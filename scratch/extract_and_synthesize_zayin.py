import os
import sys
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\.env", override=True)

# Add project root to sys.path so we can import pipeline modules
sys.path.append(r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline")

from pipeline.config import PipelineConfig
from pipeline.factory import create_tts_engine

def main():
    transcript_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\output\Yoreh_Deah_Siman_94_gemini-3_1-pro-preview_transcript.txt"
    extracted_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\output\Yoreh_Deah_Siman_94_seif_zayin_extracted.txt"
    output_audio_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\output\Yoreh_Deah_Siman_94_seif_zayin.mp3"
    
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found at {transcript_path}")
        sys.exit(1)
        
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
        
    # Extract Se'if Zayin
    start_marker = "בסעיף זין"
    end_marker = "בסעיף חית"
    
    start_idx = transcript.find(start_marker)
    if start_idx == -1:
        print("Error: Could not find start marker 'בסעיף זין' in transcript.")
        sys.exit(1)
        
    end_idx = transcript.find(end_marker, start_idx)
    if end_idx == -1:
        print("Error: Could not find end marker 'בסעיף חית' in transcript.")
        sys.exit(1)
        
    extracted_text = transcript[start_idx:end_idx].strip()
    
    print(f"Extracted Se'if Zayin text:\n{'-'*40}\n{extracted_text}\n{'-'*40}")
    
    # Save extracted text
    with open(extracted_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    print(f"Saved extracted text to {extracted_path}")
    
    # Load config and create TTS engine
    config_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\config.yaml"
    config = PipelineConfig(config_path)
    config.tts_engine = "elevenlabs"
    
    # Ensure api key is loaded from env
    if not config.elevenlabs_api_key:
        config.elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")
    
    output_audio_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\output\Yoreh_Deah_Siman_94_seif_zayin_elevenlabs.mp3"
    
    print(f"Initializing TTS engine '{config.tts_engine}' (Voice ID: {config.elevenlabs_settings.get('voice_id')})...")
    tts_engine = create_tts_engine(config)
    
    print(f"Synthesizing audio to {output_audio_path}...")
    tts_engine.synthesize(text=extracted_text, output_path=output_audio_path)
    print("Synthesis complete!")

if __name__ == "__main__":
    main()
