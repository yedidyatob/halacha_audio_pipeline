"""
Run the pronunciation stage on an existing text file, exactly as the pipeline does:
dictionary nikkud (pipeline/apply_nikkud.py) first, then the LLM adds nikkud to
context-ambiguous words ('pronunciation' settings and pronunciation_instruction in config.yaml).

Usage (from the repo root):
  python scripts/vocalize_pronunciation.py output/<file>.txt [--output out.txt]

Writes:
  <output>                  the vocalized script (default: <input stem>_vocalized.txt next to the input)
  <output stem>_changes.txt   one 'original<TAB>vocalized' line per word the LLM changed
Runs regardless of pronunciation.enabled (calling the script is the explicit request).
If the stage fails (LLM error, or output rejected by validation) nothing is
written and the script exits with code 1.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def default_output_path(input_path: str) -> str:
    stem, _ = os.path.splitext(input_path)
    return f"{stem}_vocalized.txt"


def changes_path_for(output_path: str) -> str:
    stem, _ = os.path.splitext(output_path)
    return f"{stem}_changes.txt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Add nikkud to context-ambiguous words in a text file.")
    parser.add_argument("input", help="Path to an existing .txt file (e.g. a polished script or transcript)")
    parser.add_argument("--output", default=None, help="Output path (default: <input>_vocalized.txt)")
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
    from pipeline.pronunciation import create_pronunciation_vocalizer, format_changes, PronunciationError
    from pipeline.apply_nikkud import apply_nikkud_to_abbreviations

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    with open(args.input, "r", encoding="utf-8") as f:
        text = apply_nikkud_to_abbreviations(f.read())  # dictionary nikkud first, as in the pipeline

    config = PipelineConfig(args.config)
    vocalizer = create_pronunciation_vocalizer(config)
    print(f"Vocalizing {args.input} ({len(text)} chars) via {config.pronunciation_engine} / {vocalizer.generator.model_name}...")

    try:
        vocalized, changes = vocalizer.vocalize(text)
    except PronunciationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(vocalized)
    changes_path = changes_path_for(args.output)
    with open(changes_path, "w", encoding="utf-8") as f:
        f.write(format_changes(changes))

    print(f"Vocalized script saved to: {args.output}")
    print(f"Changed words saved to:    {changes_path}")
    print(f"Words vocalized: {len(changes)}")


if __name__ == "__main__":
    main()
