import os
import sys
import re
from pipeline.config import PipelineConfig

def main():
    config_path = "config.yaml"
    analysis_path = r"C:\Users\yedidyat\.gemini\antigravity\brain\b97afda9-7234-4faa-98ca-2c9f7d58eaa5\transcript_analysis.md"
    script_output_path = "debug/siman_94_script_polished.txt"
    audio_output_path = "debug/siman_94_eleven_polished.mp3"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(analysis_path):
        print(f"Error: Analysis file '{analysis_path}' not found.")
        sys.exit(1)
        
    # Read the analysis file and extract the text block
    with open(analysis_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find the text block inside ```text ... ```
    match = re.search(r"```text\n(.*?)\n```", content, re.DOTALL)
    if not match:
        print("Error: Could not find the ```text script block in transcript_analysis.md.")
        sys.exit(1)
        
    script_text = match.group(1).strip()
    
    # Save the polished script
    os.makedirs("debug", exist_ok=True)
    with open(script_output_path, "w", encoding="utf-8") as f:
        f.write(script_text)
    print(f"Extracted polished script ({len(script_text)} characters) and saved to '{script_output_path}'.")
    
    # Load configuration and initialize ElevenLabs
    config = PipelineConfig(config_path)
    
    # Ensure ElevenLabs is the configured engine
    config.tts_engine = "elevenlabs"
    
    try:
        print(f"Synthesizing polished audio via ElevenLabs to '{audio_output_path}'...")
        tts_engine = config.get_tts_engine()
        tts_engine.synthesize(text=script_text, output_path=audio_output_path)
        
        print("\n--- ElevenLabs TTS Synthesis Success ---")
        print(f"Successfully generated polished audio lesson!")
        print(f"Saved to: '{audio_output_path}'")
        print(f"File size: {os.path.getsize(audio_output_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\n[ERROR] ElevenLabs synthesis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
