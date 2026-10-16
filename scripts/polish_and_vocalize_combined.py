"""
Experiment: Stage 3 polish AND pronunciation nikkud in ONE LLM call, to compare cost and
quality with the pipeline's two calls (polish, then the pronunciation nikkud stage).

Usage (from the repo root):
  python scripts/polish_and_vocalize_combined.py output/drafts/<draft>.txt output/relations/<relations>.txt
         [--output out.txt]

The call uses the generator configured for polish (generator.engine / model) and the same
user prompt the pipeline builds for polish (BaseScriptGenerator.polish). System instruction:

  polishing_instruction
  + experiment_combined_polish_vocalize.addendum_header
  + pronunciation_instruction (always_vocalize / never_vocalize filled) WITHOUT its first paragraph.

That first paragraph is the vocalize-only role ("return the script exactly as is, word for
word" / "the only allowed change is adding nikkud"). It conflicts with polishing, which
rewrites the text, so it is dropped as a whole paragraph and replaced by the header, which
keeps the polishing rules in force and restates the nikkud-only, no-markup rule. The script
refuses to run if that paragraph does not look as expected (it must contain the exact-copy
line "החזר את התסריט"), so a reworded instruction can't silently lose its rules.

thinking_level: pronunciation.thinking_level is used if the polish generator supports it
(Gemini); otherwise the model default is used. The script prints which one.

Validation: the output must be non-empty and contain no markup (<...>). There is no
"text unchanged" check, since polishing rewrites the text. On failure it exits 1, writing nothing.

Dictionary nikkud (pipeline/apply_nikkud.py) is applied to the result afterwards, so the
output is what the pipeline would send to TTS (in the two-call pipeline it runs between
polish and the LLM stage; here there is no separate polish output to run it on). It never
touches words the LLM already vocalized.

Writes <output> (default <draft stem>_combined_vocalized.txt next to the draft) and
<output stem>_changes.txt: 'plain<TAB>vocalized' per word the LLM vocalized (before the
dictionary pass), comparable with the changes file of scripts/vocalize_pronunciation.py.
Prints token usage and an estimated cost (prices from experiment_pricing in config.yaml).
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXACT_COPY_MARKER = "החזר את התסריט"


def read_inputs(draft_path: str, relations_path: str) -> tuple[str, str]:
    """Reads the Stage 1 draft and Stage 2 relations files; exits 1 if either is missing."""
    texts = []
    for path in (draft_path, relations_path):
        if not os.path.exists(path):
            print(f"Input file not found: {path}", file=sys.stderr)
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
    return texts[0], texts[1]


def pronunciation_rules(pronunciation_instruction: str) -> str:
    """pronunciation_instruction without its first paragraph (the exact-copy / nikkud-only lines)."""
    first, sep, rest = pronunciation_instruction.strip().partition("\n\n")
    if not sep or EXACT_COPY_MARKER not in first:
        raise ValueError(
            f"pronunciation_instruction must start with a paragraph (role and the exact-copy rule "
            f"'{EXACT_COPY_MARKER}...') followed by a blank line; cannot build the combined addendum."
        )
    return rest.strip()


def build_combined_instruction(polishing_instruction: str, addendum_header: str,
                               pronunciation_instruction: str) -> str:
    return (
        polishing_instruction.rstrip()
        + "\n\n" + addendum_header.strip()
        + "\n" + pronunciation_rules(pronunciation_instruction)
        + "\n"
    )


def validate_combined_output(text: str) -> list:
    """Returns (plain, vocalized) per vocalized word; raises PronunciationError on empty output or markup."""
    from pipeline.pronunciation import PronunciationError, find_markup, has_nikkud, strip_nikkud
    if not text or not text.strip():
        raise PronunciationError("combined call returned empty output")
    markup = find_markup(text)
    if markup:
        raise PronunciationError(f"markup in output: {markup[:5]}")
    return [(strip_nikkud(w), w) for w in text.split() if has_nikkud(w)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Polish + pronunciation-vocalize a Stage 1 draft in one LLM call.")
    parser.add_argument("draft", help="Stage 1 draft file (output/drafts/...)")
    parser.add_argument("relations", help="Stage 2 relations file (output/relations/...)")
    parser.add_argument("--output", default=None, help="Output path (default: <draft stem>_combined_vocalized.txt)")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: <repo>/config.yaml)")
    args = parser.parse_args(argv)
    args.draft = os.path.abspath(args.draft)
    args.relations = os.path.abspath(args.relations)
    args.output = (os.path.abspath(args.output) if args.output
                   else f"{os.path.splitext(args.draft)[0]}_combined_vocalized.txt")
    args.config = os.path.abspath(args.config) if args.config else str(REPO_ROOT / "config.yaml")
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=REPO_ROOT.parent / ".env")  # .env lives in the folder above the repo

    from pipeline.config import PipelineConfig
    from pipeline.factory import create_generator_engine
    from pipeline.pronunciation import PronunciationError, format_changes
    from pipeline.apply_nikkud import apply_nikkud_to_abbreviations
    from pipeline.usage import format_usage_report
    from scripts.vocalize_pronunciation import changes_path_for

    draft, relations = read_inputs(args.draft, args.relations)
    config = PipelineConfig(args.config)
    instruction = build_combined_instruction(
        config.polishing_instruction, config.experiment_combined_addendum_header, config.pronunciation_instruction
    )
    generator = create_generator_engine(config)
    pricing = config.experiment_pricing(generator.model_name)  # fail before paying for a call
    thinking_level = config.pronunciation_thinking_level if generator.supports_thinking_level else None
    print(f"Combined polish + nikkud of {args.draft} via {config.generator_engine} / {generator.model_name} "
          f"(thinking level: {thinking_level or 'model default'})...")

    result = generator.polish(draft, relations, instruction, thinking_level)
    print(format_usage_report("Combined polish + nikkud", generator.model_name, result.usage, pricing))
    try:
        changes = validate_combined_output(result.text)
    except PronunciationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        sys.exit(1)

    text = apply_nikkud_to_abbreviations(result.text.replace("\r\n", "\n"))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    changes_path = changes_path_for(args.output)
    with open(changes_path, "w", encoding="utf-8") as f:
        f.write(format_changes(changes))
    print(f"Combined script saved to: {args.output}")
    print(f"Vocalized words saved to: {changes_path}")
    print(f"Words vocalized by the LLM: {len(changes)}")


if __name__ == "__main__":
    main()
