import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\.env", override=True)

# Add project root to sys.path so we can import pipeline modules
sys.path.append(r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline")

from pipeline.config import PipelineConfig
from pipeline.factory import create_tts_engine

def main():
    config_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\config.yaml"
    text_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\output\Yoreh_Deah_Siman_94_seif_zayin_extracted.txt"
    output_audio_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\output\Yoreh_Deah_Siman_94_seif_zayin_gemini_puck_warm.mp3"
    
    if not os.path.exists(text_path):
        print(f"Error: Text file not found at {text_path}")
        sys.exit(1)
        
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print(f"Loaded text ({len(text)} characters).")
    
    # Load config and override engine to gemini (which now uses GCP internally)
    config = PipelineConfig(config_path)
    config.tts_engine = "gemini"
    
    print("Initializing Gemini TTS engine (GCP)...")
    tts_engine = create_tts_engine(config)
    
    # Override voice to Puck
    tts_engine.voice_name = "Puck"
    
    # Supply style prompt natively via the input.prompt JSON property
    tts_engine.prompt = "Read aloud in a warm, deductive tone."
    
    print(f"Synthesizing to {output_audio_path}...")
    tts_engine.synthesize(text=text, output_path=output_audio_path)
    print("Synthesis complete!")

if __name__ == "__main__":
    main()
