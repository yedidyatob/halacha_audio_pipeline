import os
import sys
import json
import base64
import requests
import logging

import google.auth
from google.auth.transport.requests import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GCP_TTS_Test")

def main():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\.env", override=True)
    
    # Attempt to get Google Cloud credentials
    try:
        credentials, project = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        credentials.refresh(Request())
        token = credentials.token
    except Exception as e:
        logger.error(f"Failed to load Google Cloud credentials: {e}")
        logger.info("Ensure GOOGLE_APPLICATION_CREDENTIALS is set in your environment.")
        return

    url = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
        # Request a quota project to ensure billing is routed if required
        "x-goog-user-project": project if project else "" 
    }

    payload = {
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "pitch": 0,
            "speakingRate": 1
        },
        "input": {
            "text": "מכאן אנו עוברים לסעיף זין, ועוברים מנפילה פסיבית לפעולה אקטיבית של חיתוך. המחבר פוסק שבשר רותח שחתכו בסכין חלבית, כל החתיכה אסורה אם אין בה שישים כנגד מקום הסכין שחתך. אבל אם הסכין אינו בן יומו, אינו אוסר אלא כדי קליפה. הרמ\"א מוסיף שכל זה בבשר רותח בכלי ראשון, ואז אם הסכין בן יומו ואין שישים, הכל אסור ואף הסכין צריך הגעלה."
        },
        "voice": {
            "languageCode": "he-il",
            "modelName": "gemini-3.1-flash-tts-preview",
            "name": "Achird"
        }
    }

    logger.info("Sending JSON payload to Google Cloud TTS API (v1beta1)...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        logger.error("Request timed out after 60 seconds.")
        return

    if response.status_code != 200:
        logger.error(f"API Error ({response.status_code}): {response.text}")
        return

    data = response.json()
    audio_content = data.get("audioContent")
    
    if not audio_content:
        logger.error("No audioContent found in response.")
        return

    pcm_data = base64.b64decode(audio_content)
    logger.info(f"Received {len(pcm_data)} bytes of LINEAR16 audio.")

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'test_audio')
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "gcp_tts_test.wav")

    # Write as a standard WAV file since it's LINEAR16 (PCM)
    import wave
    with wave.open(audio_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(24000) # Assuming 24kHz
        wav_file.writeframes(pcm_data)

    logger.info(f"Audio saved to: {audio_path}")

if __name__ == "__main__":
    main()
