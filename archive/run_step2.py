import os
import sys
from pipeline.config import PipelineConfig
from pipeline.generator import GeminiScriptGenerator

def main():
    config_path = "config.yaml"
    context_path = "debug/siman_94_context.txt"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(context_path):
        print(f"Error: Context file '{context_path}' not found. Please run Step 1 first.")
        sys.exit(1)
        
    config = PipelineConfig(config_path)
    
    # Check if Gemini API Key is available
    if not config.gemini_api_key:
        print("\n[ERROR] Gemini API key not found.")
        print("Please set your API key as an environment variable before running:")
        print("PowerShell: $env:GEMINI_API_KEY='your_key'")
        print("Command Prompt: set GEMINI_API_KEY=your_key")
        sys.exit(1)
        
    with open(context_path, "r", encoding="utf-8") as f:
        master_context = f.read()
        
    generator = GeminiScriptGenerator(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model_name,
        temperature=config.gemini_temperature
    )
    
    print("Calling Gemini API to generate TTS-optimized Hebrew script for Siman 94...")
    try:
        script_text = generator.generate_siman_script(
            siman=94,
            master_context=master_context,
            system_instruction=config.gemini_system_instruction
        )
        
        # Save output
        os.makedirs("debug", exist_ok=True)
        script_path = "debug/siman_94_script.txt"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_text)
            
        print("\n--- Gemini Script Generation Stats ---")
        char_count = len(script_text)
        word_count = len(script_text.split())
        print(f"* Characters: {char_count}")
        print(f"* Words: {word_count}")
        print(f"Script saved to '{script_path}'")
        
        # Estimate ElevenLabs pricing
        # ElevenLabs charges $15 for 100,000 characters (Creator plan) or $0.30 per 1,000 characters on pay-as-you-go, or $0.00 for the free tier (10,000 characters per month).
        # Let's show detailed pricing based on character count:
        est_cost_free = 0.0
        est_cost_paid = (char_count / 1000) * 0.18 # ~$0.18 per 1,000 characters on Creator tier
        
        print("\nEstimated ElevenLabs Cost:")
        print(f"- Characters to synthesize: {char_count}")
        print(f"- On Free Tier: Free (uses {char_count}/10,000 monthly characters)")
        print(f"- On Paid Creator Tier (~$0.18 per 1k chars): ~${est_cost_paid:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
