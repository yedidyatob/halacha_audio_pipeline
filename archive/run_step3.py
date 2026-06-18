import os
import sys
from pipeline.config import PipelineConfig

def main():
    config_path = "config.yaml"
    script_path = "debug/siman_94_script.txt"
    output_audio_path = "debug/siman_94.mp3"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(script_path):
        print(f"Error: Script file '{script_path}' not found. Please run Step 2 first.")
        sys.exit(1)
        
    config = PipelineConfig(config_path)
    
    # Read the script
    with open(script_path, "r", encoding="utf-8") as f:
        script_text = f.read()
        
    char_count = len(script_text)
    print(f"Preparing to synthesize {char_count} characters using ElevenLabs...")
    
    # Get TTS engine
    try:
        tts_engine = config.get_tts_engine()
        
        # Call ElevenLabs
        tts_engine.synthesize(text=script_text, output_path=output_audio_path)
        
        print("\n--- ElevenLabs TTS Synthesis Success ---")
        print(f"Successfully generated audio lesson!")
        print(f"Saved to: '{output_audio_path}'")
        print(f"File size: {os.path.getsize(output_audio_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\n[ERROR] ElevenLabs synthesis failed: {e}")
        print("\nIf you are on the ElevenLabs Free Tier and hit a character limit:")
        print("1. Check your remaining monthly character quota on the ElevenLabs dashboard.")
        print("2. You can try a shorter script or select a different model.")
        print("3. Alternatively, you can switch to Google Cloud TTS in config.yaml which is more cost-effective.")
        sys.exit(1)

if __name__ == "__main__":
    main()
