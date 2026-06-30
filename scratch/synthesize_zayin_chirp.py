import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\.env", override=True)

# Add project root to sys.path so we can import pipeline modules
sys.path.append(r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline")

from pipeline.tts import GoogleCloudTTS

def main():
    text_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\output\Yoreh_Deah_Siman_94_seif_zayin_extracted.txt"
    output_audio_path = r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\output\Yoreh_Deah_Siman_94_seif_zayin_he_IL_Wavenet_D.mp3"
    
    if not os.path.exists(text_path):
        print(f"Error: Text file not found at {text_path}")
        sys.exit(1)
        
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    print(f"Loaded text ({len(text)} characters).")
    
    print("Initializing Google Cloud TTS engine (Chirp)...")
    
    # Use GoogleCloudTTS with the requested voice
    tts_engine = GoogleCloudTTS(
        voice_name="he-IL-Wavenet-D",
        language_code="he-IL"
    )
    
    print(f"Synthesizing to {output_audio_path}...")
    tts_engine.synthesize(text=text, output_path=output_audio_path)
    print("Synthesis complete!")

if __name__ == "__main__":
    main()
