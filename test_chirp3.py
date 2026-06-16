import os
import sys
from google.cloud import texttospeech
from pipeline.config import PipelineConfig

def main():
    config = PipelineConfig("config.yaml")
    if not config.google_tts_credentials or not os.path.exists(config.google_tts_credentials):
        print("Error: Google credentials not configured.")
        sys.exit(1)
        
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.google_tts_credentials
    client = texttospeech.TextToSpeechClient()
    
    # Try synthesizing a small test using he-IL-Chirp3-HD-Achird
    text = "שלום, זהו שיעור הלכה בסימן צד."
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Configure Chirp 3 HD voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="he-IL",
        name="he-IL-Chirp3-HD-Achird"
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    
    print("Testing synthesis with he-IL-Chirp3-HD-Achird...")
    try:
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        output_file = "debug/test_chirp3.mp3"
        os.makedirs("debug", exist_ok=True)
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        print(f"Success! Written to {output_file}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
