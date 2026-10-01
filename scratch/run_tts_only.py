"""
One-off script: synthesize TTS using ElevenLabs on an existing transcript file.
Usage: python scratch/run_tts_only.py <transcript_path> [--config config.yaml]
"""
import sys
import os
import argparse
from dotenv import load_dotenv

from pathlib import Path

# Get the path to the directory where the current script resides
current_dir = Path(__file__).resolve().parent

# Build the path to the .env file in the parent folder of the directory root
dotenv_path = current_dir.parent.parent / ".env"

# Load the file
load_dotenv(dotenv_path=dotenv_path)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.config import PipelineConfig
from pipeline.utils import save_output_file
from pipeline.logger import get_logger

logger = get_logger("run_tts_only")

def main():
    parser = argparse.ArgumentParser(description="Run ElevenLabs TTS on an existing transcript file.")
    parser.add_argument("transcript", help="Path to the transcript .txt file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--voice-id", default=None, help="Override ElevenLabs voice ID")
    parser.add_argument("--voice-name", default=None, help="Label used in output filename (e.g. 'Will')")
    args = parser.parse_args()

    if not os.path.exists(args.transcript):
        logger.error(f"Transcript file not found: {args.transcript}")
        sys.exit(1)

    with open(args.transcript, "r", encoding="utf-8") as f:
        text = f.read()
    logger.info(f"Loaded transcript: {args.transcript} ({len(text)} chars)")

    config = PipelineConfig(args.config)

    # Force ElevenLabs regardless of what's in config
    from pipeline.tts import ElevenLabsTTS
    el_settings = config.elevenlabs_settings
    voice_id = args.voice_id or el_settings["voice_id"]
    tts_engine = ElevenLabsTTS(
        api_key=config.elevenlabs_api_key if hasattr(config, 'elevenlabs_api_key') else None,
        voice_id=voice_id,
        model_id=el_settings["model_id"],
        stability=el_settings["stability"],
        similarity_boost=el_settings["similarity_boost"],
        ssl_verify=config.ssl_verify,
        chunk_gap_paragraph_ms=config.tts_chunk_gap_paragraph_ms,
        chunk_gap_sentence_ms=config.tts_chunk_gap_sentence_ms,
    )
    logger.info(f"Using ElevenLabs voice ID: {voice_id}")

    # Derive output base name from transcript filename
    base = os.path.splitext(os.path.basename(args.transcript))[0]
    # Remove "_transcript" suffix if present
    base = base.replace("_transcript", "")
    voice_label = args.voice_name if args.voice_name else "elevenlabs"
    output_base = f"{base}_elevenlabs_{voice_label}"

    def tts_callback(path: str) -> None:
        logger.info(f"Synthesizing to: {path} ...")
        tts_engine.synthesize(text=text, output_path=path)

    save_output_file(
        directory=config.output_dir,
        base_name=output_base,
        extension="mp3",
        content=None,
        logger=logger,
        write_callback=tts_callback
    )
    logger.info("TTS synthesis completed successfully.")

if __name__ == "__main__":
    main()
