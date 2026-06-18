import os
import sys
import wave
from google import genai
from google.genai import types
from pipeline.config import PipelineConfig

def main():
    config_path = "config.yaml"
    context_path = "debug/siman_94_context.txt"
    output_wav_path = "debug/siman_94_gemini_native.wav"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(context_path):
        print(f"Error: Context file '{context_path}' not found. Please run Step 1 first.")
        sys.exit(1)
        
    config = PipelineConfig(config_path)
    
    if not config.gemini_api_key:
        print("Error: Gemini API key not found.")
        sys.exit(1)
        
    client = genai.Client(api_key=config.gemini_api_key)
    
    # We will use gemini-2.5-flash-preview-tts which is verified to support native audio generation
    model_name = "gemini-2.5-flash-preview-tts"
    
    with open(context_path, "r", encoding="utf-8") as f:
        master_context = f.read()
        
    print(f"Requesting native audio generation from {model_name}...")
    print("Using Voice: Puck (Upbeat Male)")
    
    # Configure the native audio generation request
    generate_config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Puck"
                )
            )
        ),
        system_instruction=config.gemini_system_instruction,
        temperature=0.3
    )
    
    # Ask for the lesson directly. The model will generate the audio directly
    user_prompt = (
        f"להלן קובץ המקורות המלא. אנא הפק כעת שיעור שמע (אודיו) מפורט ומעמיק במיוחד עבור סימן צ\"ד.\n\n"
        f"דרישות קריטיות להשלמה:\n"
        f"1. עליך להקיף את כל הסעיפים והדעות המופיעים בסימן צ\"ד עצמו. אל תסכם או תדלג על אף שיטה או קושיה.\n"
        f"2. דבר כפסקאות רצופות, ללא עיצובי מרקדאון, פתח את כל ראשי התיבות, ואל תשתמש בסוגריים.\n\n"
        f"קובץ המקורות:\n"
        f"{master_context}"
    )
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=generate_config
        )
        
        audio_part = None
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    audio_part = part
                    break
                    
        if audio_part and audio_part.inline_data:
            pcm_data = audio_part.inline_data.data
            mime_type = audio_part.inline_data.mime_type
            print(f"\nSuccess! Audio received. Mime-type: {mime_type}")
            print(f"PCM Data length: {len(pcm_data)} bytes")
            
            # The API returns raw 24kHz PCM (Mono, 16-bit). Let's wrap it in a WAV container.
            print(f"Writing audio to WAV file: '{output_wav_path}'...")
            
            os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
            with wave.open(output_wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)      # Mono
                wav_file.setsampwidth(2)      # 16-bit (2 bytes)
                wav_file.setframerate(24000)  # 24kHz sample rate
                
                # In case data is base64 string (should be bytes in genai SDK, but let's handle both)
                if isinstance(pcm_data, str):
                    import base64
                    pcm_data = base64.b64decode(pcm_data)
                    
                wav_file.writeframes(pcm_data)
                
            file_size_mb = os.path.getsize(output_wav_path) / 1024 / 1024
            # Calculate duration: 24000 frames per second, 2 bytes per frame, 1 channel
            # Bytes per second = 24000 * 2 * 1 = 48000 bytes/sec
            duration_seconds = len(pcm_data) / 48000
            
            print(f"\n--- Native Audio Generation Success ---")
            print(f"* Saved to: '{output_wav_path}'")
            print(f"* File Size: {file_size_mb:.2f} MB")
            print(f"* Duration: {duration_seconds / 60:.2f} minutes ({int(duration_seconds)} seconds)")
            
        else:
            print("\nError: No audio data returned by the model.")
            if response.text:
                print(f"Text response: {response.text}")
                
    except Exception as e:
        print(f"\n[ERROR] Native audio generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
