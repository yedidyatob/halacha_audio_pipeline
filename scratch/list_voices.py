import os
from google.cloud import texttospeech
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\.env", override=True)

client = texttospeech.TextToSpeechClient()
voices = client.list_voices()

print("Available Hebrew (he-IL) voices:")
for voice in voices.voices:
    for language_code in voice.language_codes:
        if "he" in language_code.lower():
            print(f"Name: {voice.name}")
            print(f"  Language: {language_code}")
            print(f"  SSML Gender: {voice.ssml_gender.name}")
            print(f"  Rate: {voice.natural_sample_rate_hertz} Hz")
            print("-" * 40)
