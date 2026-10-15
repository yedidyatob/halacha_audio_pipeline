"""
Experiment: Stage 3 polish AND pronunciation tagging in ONE LLM call, to compare cost and
quality with the pipeline's two calls (polish, then tag).

Usage (from the repo root):
  python scripts/polish_and_tag_combined.py output/drafts/<draft>.txt output/relations/<relations>.txt
         [--output out.txt] [--apply-nikkud]

The call uses the generator configured for polish (generator.engine / model) and the same
user prompt the pipeline builds for polish (BaseScriptGenerator.polish). System instruction:

  polishing_instruction
  + experiment_combined_polish_tag.addendum_header
  + pronunciation_instruction (always_tag / never_tag filled) WITHOUT its first paragraph.

That first paragraph is the tagging-only role ("return the script exactly as is, word for
word" / "the only allowed change is wrapping words in tags" + the tag format). It conflicts
with polishing, which rewrites the text, so it is dropped as a whole paragraph and replaced by
the header, which keeps the polishing rules in force and restates the tag format. The script
refuses to run if that paragraph does not look as expected (it must contain the <phoneme
format line), so a reworded instruction can't silently lose its rules.

thinking_level: pronunciation.thinking_level is used if the polish generator supports it
(Gemini); otherwise the model default is used. The script prints which one.

Validation: phoneme tags must be well formed (the "text unchanged outside tags" check does
not apply because the text is newly written). Malformed or empty output exits 1, writing nothing.

Writes <output> (default <draft stem>_combined_tagged.txt next to the draft) and
<output stem>_tags.txt (word<TAB>ipa per tag). Prints token usage and an estimated cost
(prices from experiment_pricing in config.yaml).
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    """pronunciation_instruction without its first paragraph (the exact-copy / tag-only lines)."""
    first, sep, rest = pronunciation_instruction.strip().partition("\n\n")
    if not sep or "<phoneme" not in first:
        raise ValueError(
            "pronunciation_instruction must start with a paragraph (role, exact-copy rule and the "
            "<phoneme ...> format line) followed by a blank line; cannot build the combined addendum."
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
    """Returns the (word, ipa) pairs; raises PronunciationTaggingError on empty output or malformed tags."""
    from pipeline.pronunciation import PronunciationTaggingError, extract_phoneme_pairs, find_malformed_tags
    if not text or not text.strip():
        raise PronunciationTaggingError("combined call returned empty output")
    malformed = find_malformed_tags(text)
    if malformed:
        raise PronunciationTaggingError(f"malformed phoneme tag(s): {malformed[:5]}")
    return extract_phoneme_pairs(text)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Polish + pronunciation-tag a Stage 1 draft in one LLM call.")
    parser.add_argument("draft", help="Stage 1 draft file (output/drafts/...)")
    parser.add_argument("relations", help="Stage 2 relations file (output/relations/...)")
    parser.add_argument("--output", default=None, help="Output path (default: <draft stem>_combined_tagged.txt)")
    parser.add_argument("--apply-nikkud", action="store_true",
                        help="Also apply nikkud after tagging (matches what the pipeline sends to TTS)")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: <repo>/config.yaml)")
    args = parser.parse_args(argv)
    args.draft = os.path.abspath(args.draft)
    args.relations = os.path.abspath(args.relations)
    args.output = (os.path.abspath(args.output) if args.output
                   else f"{os.path.splitext(args.draft)[0]}_combined_tagged.txt")
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
    from pipeline.pronunciation import PronunciationTaggingError, format_pairs
    from pipeline.apply_nikkud import apply_nikkud_to_abbreviations
    from pipeline.usage import format_usage_report
    from scripts.tag_pronunciation import tags_path_for

    draft, relations = read_inputs(args.draft, args.relations)
    config = PipelineConfig(args.config)
    instruction = build_combined_instruction(
        config.polishing_instruction, config.experiment_combined_addendum_header, config.pronunciation_instruction
    )
    generator = create_generator_engine(config)
    pricing = config.experiment_pricing(generator.model_name)  # fail before paying for a call
    thinking_level = config.pronunciation_thinking_level if generator.supports_thinking_level else None
    print(f"Combined polish + tagging of {args.draft} via {config.generator_engine} / {generator.model_name} "
          f"(thinking level: {thinking_level or 'model default'})...")

    result = generator.polish(draft, relations, instruction, thinking_level)
    print(format_usage_report("Combined polish + tagging", generator.model_name, result.usage, pricing))
    try:
        pairs = validate_combined_output(result.text)
    except PronunciationTaggingError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        sys.exit(1)

    text = result.text.replace("\r\n", "\n")
    if args.apply_nikkud:
        text = apply_nikkud_to_abbreviations(text)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    tags_path = tags_path_for(args.output)
    with open(tags_path, "w", encoding="utf-8") as f:
        f.write(format_pairs(pairs))
    print(f"Combined script saved to: {args.output}")
    print(f"Tag list saved to:        {tags_path}")
    print(f"Words tagged: {len(pairs)}")


if __name__ == "__main__":
    main()
