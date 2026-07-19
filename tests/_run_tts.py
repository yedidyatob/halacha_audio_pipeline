"""
One-off helper: synthesize an existing transcript file via ElevenLabs TTS.
Usage: python tests/_run_tts.py <transcript_file> [output_mp3] [--voice-id <id>]
"""
import sys
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

# Allow imports from the project root when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.config import PipelineConfig
from pipeline.factory import create_tts_engine
from pipeline.logger import get_logger

logger = get_logger("run_tts")


def main():
    parser = argparse.ArgumentParser(
        description="Run ElevenLabs TTS on an existing transcript file."
    )
    parser.add_argument("input", help="Path to the input transcript .txt file")
    parser.add_argument("output", nargs="?", help="Output .mp3 path (optional)")
    parser.add_argument("--voice-id", default=None, help="Override ElevenLabs voice ID")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Derive output path if not provided
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join("output", base + ".mp3")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Input : {args.input} ({len(text)} chars)")
    logger.info(f"Output: {output_path}")

    config = PipelineConfig("config.yaml")
    config.tts_engine = "elevenlabs"

    tts = create_tts_engine(config)

    # Override voice ID if provided on the command line
    if args.voice_id:
        logger.info(f"Overriding voice ID: {args.voice_id}")
        tts.voice_id = args.voice_id

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tts.synthesize(text=text, output_path=output_path)
    logger.info(f"Done. Audio saved to: {output_path}")


if __name__ == "__main__":
    main()
