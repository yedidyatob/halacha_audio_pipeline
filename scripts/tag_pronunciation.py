"""
Tag an existing transcript with IPA pronunciation (phoneme) tags, using the
'pronunciation' settings and 'pronunciation_instruction' from config.yaml.

Usage (from the repo root):
  python scripts/tag_pronunciation.py output/<file>_transcript.txt [--output out.txt] [--apply-nikkud]

Writes:
  <output>              the tagged script (default: <input stem>_tagged.txt next to the input)
  <output stem>_tags.txt  one 'word<TAB>ipa' line per tagged word
Runs regardless of pronunciation.enabled (calling the script is the explicit request).
If tagging fails (LLM error, text changed outside tags, malformed tags) nothing is written
and the script exits with code 1.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def default_output_path(input_path: str) -> str:
    stem, _ = os.path.splitext(input_path)
    return f"{stem}_tagged.txt"


def tags_path_for(output_path: str) -> str:
    stem, _ = os.path.splitext(output_path)
    return f"{stem}_tags.txt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Add IPA phoneme tags to an existing transcript.")
    parser.add_argument("input", help="Path to an existing transcript .txt file")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_tagged.txt)")
    parser.add_argument("--apply-nikkud", action="store_true",
                        help="Also apply nikkud after tagging (matches what the pipeline sends to TTS)")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: <repo>/config.yaml)")
    args = parser.parse_args(argv)
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output) if args.output else default_output_path(args.input)
    args.config = os.path.abspath(args.config) if args.config else str(REPO_ROOT / "config.yaml")
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)  # config.yaml / output paths are repo-relative

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=REPO_ROOT.parent / ".env")  # .env lives in the folder above the repo

    from pipeline.config import PipelineConfig
    from pipeline.pronunciation import create_pronunciation_tagger, format_pairs, PronunciationTaggingError
    from pipeline.apply_nikkud import apply_nikkud_to_abbreviations

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    config = PipelineConfig(args.config)
    tagger = create_pronunciation_tagger(config)
    print(f"Tagging {args.input} ({len(text)} chars) via {config.pronunciation_engine} / {tagger.generator.model_name}...")

    try:
        tagged, pairs = tagger.tag(text)
    except PronunciationTaggingError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        sys.exit(1)
    if args.apply_nikkud:
        tagged = apply_nikkud_to_abbreviations(tagged)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(tagged)
    tags_path = tags_path_for(args.output)
    with open(tags_path, "w", encoding="utf-8") as f:
        f.write(format_pairs(pairs))

    print(f"Tagged script saved to: {args.output}")
    print(f"Tag list saved to:      {tags_path}")
    print(f"Words tagged: {len(pairs)}")


if __name__ == "__main__":
    main()
