"""
Synthesize a text file with the TTS engine configured in config.yaml (tts.engine),
using pipeline/tts.py (same chunking and chunk gaps as the pipeline).

Usage (from the repo root):
  python scripts/tts_from_file.py output/<file>.txt [--output out.mp3]

Default output: the input path with the engine's audio extension (e.g. <input stem>.mp3).
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the configured TTS engine on a text file.")
    parser.add_argument("input", help="Path to the text file to synthesize")
    parser.add_argument("--output", default=None, help="Output audio path (default: <input stem>.<ext>)")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: <repo>/config.yaml)")
    args = parser.parse_args(argv)
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output) if args.output else None
    args.config = os.path.abspath(args.config) if args.config else str(REPO_ROOT / "config.yaml")
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=REPO_ROOT.parent / ".env")  # .env lives in the folder above the repo

    from pipeline.config import PipelineConfig
    from pipeline.factory import create_tts_engine

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    config = PipelineConfig(args.config)
    tts = create_tts_engine(config)
    output = args.output or os.path.splitext(args.input)[0] + "." + tts.file_extension
    print(f"Synthesizing {args.input} ({len(text)} chars) with '{config.tts_engine}' -> {output}")
    tts.synthesize(text=text, output_path=output)
    print(f"Audio saved to: {output}")


if __name__ == "__main__":
    main()
