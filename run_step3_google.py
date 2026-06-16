import os
import sys
from pipeline.config import PipelineConfig

def main():
    config_path = "config.yaml"
    script_path = "debug/siman_94_script.txt"
    output_audio_path = "debug/siman_94_google.mp3"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(script_path):
        print(f"Error: Script file '{script_path}' not found. Please run Step 2 first.")
        sys.exit(1)
        
    config = PipelineConfig(config_path)
    
    # Check if Google TTS credentials path is set
    if not config.google_tts_credentials:
        print("\n[ERROR] Google TTS credentials path not found in config.yaml.")
        print("Please configure 'api_keys.google_tts_credentials' with the path to your JSON service account key.")
        sys.exit(1)
        
    if not os.path.exists(config.google_tts_credentials):
        print(f"\n[ERROR] Google credentials file not found at: {config.google_tts_credentials}")
        sys.exit(1)
        
    # Read the script
    with open(script_path, "r", encoding="utf-8") as f:
        script_text = f.read()
        
    char_count = len(script_text)
    print(f"Preparing to synthesize {char_count} characters using Google Cloud TTS...")
    
    # Temporarily force the engine to Google for this comparison script
    config.tts_engine = "google"
    
    try:
        tts_engine = config.get_tts_engine()
        
        # Call Google Cloud TTS
        tts_engine.synthesize(text=script_text, output_path=output_audio_path)
        
        print("\n--- Google Cloud TTS Synthesis Success ---")
        print(f"Successfully generated Google audio lesson!")
        print(f"Saved to: '{output_audio_path}'")
        print(f"File size: {os.path.getsize(output_audio_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\n[ERROR] Google Cloud TTS synthesis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
