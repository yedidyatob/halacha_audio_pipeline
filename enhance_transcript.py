"""
enhance_transcript.py

CLI entry point for Stage 3.5 — ElevenLabs Audio Tag Enhancement.

Reads a polished TTS transcript, passes it through Gemini to inject
ElevenLabs expression tags ([energetic], [thoughtful], [short pause], etc.),
and saves the result alongside the original transcript in the output/ directory.

Usage examples:
  # Enhance a specific transcript file
  python enhance_transcript.py output/Yoreh_Deah_Siman_94_gemini-3_1-pro-preview_transcript.txt

  # Enhance with a custom output directory
  python enhance_transcript.py output/Yoreh_Deah_Siman_94_gemini-3_1-pro-preview_transcript.txt --output-dir output/enhanced

  # Use a specific Gemini model
  python enhance_transcript.py output/... --model gemini-2.0-flash

  # Find and enhance the latest transcript automatically (no input path needed)
  python enhance_transcript.py --latest
"""

import os
import sys
import argparse
import glob
from dotenv import load_dotenv

load_dotenv()

from pipeline.logger import get_logger
from pipeline.enhancer import ElevenLabsEnhancer

logger = get_logger("enhance_transcript")


def find_latest_transcript(search_dir: str) -> str:
    """
    Scans search_dir for timestamped transcript files and returns the path
    of the most recently generated one.

    Timestamped transcripts follow the pattern:
        *_transcript_YYYYMMDD_HHMMSS.txt
    """
    pattern = os.path.join(search_dir, "*_transcript_[0-9]*_[0-9]*.txt")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No timestamped transcript files found in '{search_dir}'. "
            "Run the main pipeline first or provide an explicit path."
        )

    # Sort by the embedded timestamp suffix (lexicographic == chronological for YYYYMMDD_HHMMSS)
    matches.sort()
    latest = matches[-1]
    logger.info(f"Auto-detected latest transcript: {latest}")
    return latest


def derive_output_base_name(input_path: str) -> str:
    """
    Converts a transcript filename into the enhanced output base name.

    Examples:
      Yoreh_Deah_Siman_94_gemini-3_1-pro-preview_transcript_20260712_234539.txt
        -> Yoreh_Deah_Siman_94_gemini-3_1-pro-preview_enhanced
      Yoreh_Deah_Siman_94_gemini-3_1-pro-preview_transcript.txt
        -> Yoreh_Deah_Siman_94_gemini-3_1-pro-preview_enhanced
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]  # drop .txt

    # Strip trailing timestamp  _YYYYMMDD_HHMMSS  if present
    import re
    stem = re.sub(r"_\d{8}_\d{6}$", "", stem)

    # Replace _transcript suffix with _enhanced
    if stem.endswith("_transcript"):
        stem = stem[: -len("_transcript")] + "_enhanced"
    else:
        stem = stem + "_enhanced"

    return stem


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3.5 — ElevenLabs Enhancement: inject audio expression tags "
            "into a polished Hebrew Halachic TTS transcript."
        )
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "input_path",
        nargs="?",
        help="Path to the transcript .txt file to enhance.",
    )
    input_group.add_argument(
        "--latest",
        action="store_true",
        help="Automatically find and enhance the most recently generated transcript in --search-dir.",
    )

    parser.add_argument(
        "--search-dir",
        type=str,
        default="output",
        help="Directory to search when using --latest (default: output).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write the enhanced transcript into. "
            "Defaults to the same directory as the input file."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-flash-lite",
        help="Gemini model to use for enhancement (default: gemini-3.1-flash-lite).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature (default: 0.2 — high fidelity to source text).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # ------------------------------------------------------------------ #
    # 1. Resolve input file
    # ------------------------------------------------------------------ #
    if args.latest:
        try:
            input_path = find_latest_transcript(args.search_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
    else:
        input_path = args.input_path
        if not os.path.isfile(input_path):
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Read transcript
    # ------------------------------------------------------------------ #
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
        logger.info(f"Read transcript: {input_path} ({len(transcript_text)} chars)")
    except Exception as e:
        logger.error(f"Failed to read transcript: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3. Resolve output directory and base name
    # ------------------------------------------------------------------ #
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(input_path))
    base_name = derive_output_base_name(input_path)

    logger.info(f"Output directory : {output_dir}")
    logger.info(f"Output base name : {base_name}")

    # ------------------------------------------------------------------ #
    # 4. Initialise enhancer
    # ------------------------------------------------------------------ #
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    try:
        enhancer = ElevenLabsEnhancer(
            api_key=api_key,
            model_name=args.model,
            temperature=args.temperature,
        )
    except Exception as e:
        logger.error(f"Failed to initialise enhancer: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 5. Enhance and save
    # ------------------------------------------------------------------ #
    try:
        history_path = enhancer.enhance_and_save(
            transcript_text=transcript_text,
            output_dir=output_dir,
            base_name=base_name,
        )
        logger.info(f"Enhanced transcript saved to: {history_path}")
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        sys.exit(1)

    logger.info("Stage 3.5 complete.")


if __name__ == "__main__":
    main()
