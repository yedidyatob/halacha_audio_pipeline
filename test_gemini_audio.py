import os
import sys
from google import genai
from google.genai import types
from pipeline.config import PipelineConfig

def main():
    config = PipelineConfig("config.yaml")
    if not config.gemini_api_key:
        print("Error: Gemini API key not found.")
        sys.exit(1)
        
    client = genai.Client(api_key=config.gemini_api_key)
    model_name = "gemini-2.5-flash-preview-tts"
    
    print(f"Requesting native audio generation from {model_name}...")
    
    generate_config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Puck"
                )
            )
        ),
        temperature=0.3
    )
    
    prompt = (
        "אתה רב ומחנך. תגיד שלום ותסביר בקצרה בשתיים-שלוש שורות "
        "מה הנושא של שיעור היום (סימן צד - הלכות כף חולבת בקדרה של בשר). "
        "הקרא זאת בקול חם, אנרגטי ומזמין."
    )
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=generate_config
        )
        
        print("\n--- Full Response Object ---")
        print(f"Candidates list: {response.candidates}")
        if response.candidates:
            candidate = response.candidates[0]
            print(f"Candidate Finish Reason: {candidate.finish_reason}")
            print(f"Candidate Content: {candidate.content}")
            if hasattr(candidate, 'safety_ratings'):
                print(f"Safety Ratings: {candidate.safety_ratings}")
        else:
            print("No candidates returned.")
            
    except Exception as e:
        print(f"\nError during native audio generation: {e}")

if __name__ == "__main__":
    main()
