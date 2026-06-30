import os
import sys
import json
import base64
import requests
import logging
import lameenc
from mutagen.mp3 import MP3

# Add the pipeline module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.config import PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTS_REST_Test")

def main():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config = PipelineConfig(config_path)

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\.env", override=True)
    api_key = config.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("GEMINI_API_KEY is not set.")
        return

    test_text = (
        "מכאן אנו עוברים לסעיף זין, ועוברים מנפילה פסיבית לפעולה אקטיבית של חיתוך. המחבר פוסק "
        "שבשר רותח שחתכו בסכין חלבית, כל החתיכה אסורה אם אין בה שישים כנגד מקום הסכין שחתך. "
        "אבל אם הסכין אינו בן יומו, אינו אוסר אלא כדי קליפה. הרמ\"א מוסיף שכל זה בבשר רותח בכלי ראשון, "
        "ואז אם הסכין בן יומו ואין שישים, הכל אסור ואף הסכין צריך הגעלה. אבל אם הוא כלי שני, הבשר "
        "צריך קליפה והסכין נעיצה בקרקע. ואפילו אם הסכין אינו בן יומו, יש לקלוף את הבשר מעט משום "
        "שמנונית הסכין."
    )
    
    logger.info(f"Test Text Length: {len(test_text)} characters")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-tts-preview:generateContent?key={api_key}"
    
    prompt_content = (
        "Audio Profile: Professional male narrator. Vibe: Recording in a quiet studio. "
        "Director notes: Speak clearly and slowly in Hebrew. Do not add any introduction or comments, "
        "and do not generate any text response. Only output audio. "
        "Here is the script to read:\n\n" + test_text
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_content}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": "Achird"
                    }
                }
            }
        }
    }

    logger.info("Sending REST request to Gemini API...")
    
    # Increase timeout significantly for generation
    response = requests.post(url, json=payload, timeout=120)
    
    if response.status_code != 200:
        logger.error(f"API Error: {response.status_code} - {response.text}")
        return
        
    data = response.json()
    
    usage = data.get("usageMetadata", {})
    prompt_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)
    total_tokens = usage.get("totalTokenCount", 0)
    
    logger.info(f"Tokens Used: Input={prompt_tokens}, Output={output_tokens}, Total={total_tokens}")
    
    # Audio tokens are billed at $20.00 / 1M
    cost = (prompt_tokens * (0.075 / 1_000_000)) + (output_tokens * (20.00 / 1_000_000))
    logger.info(f"Estimated Cost: ${cost:.6f} USD")
    
    # Extract audio
    candidates = data.get("candidates", [])
    if not candidates:
        logger.error("No candidates in response.")
        return
        
    parts = candidates[0].get("content", {}).get("parts", [])
    audio_b64 = None
    for part in parts:
        if "inlineData" in part:
            audio_b64 = part["inlineData"]["data"]
            break
            
    if not audio_b64:
        logger.error("No inlineData found in response.")
        return
        
    pcm_data = base64.b64decode(audio_b64)
    logger.info(f"Received {len(pcm_data)} bytes of raw PCM audio.")
    
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(24000)
    encoder.set_channels(1)
    encoder.set_quality(2)
    
    mp3_data = encoder.encode(pcm_data)
    mp3_data += encoder.flush()
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'test_audio')
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "saif_zayin_para1_rest.mp3")
    
    with open(audio_path, "wb") as f:
        f.write(mp3_data)
        
    if os.path.exists(audio_path):
        audio_info = MP3(audio_path)
        duration_sec = audio_info.info.length
        logger.info(f"Generated Audio Duration: {duration_sec:.2f} seconds")
        if duration_sec > 0:
            tokens_per_sec = output_tokens / duration_sec
            logger.info(f"Calculated tokens per second: {tokens_per_sec:.1f}")
        logger.info(f"Calculated tokens per character: {output_tokens / len(test_text):.1f}")
        logger.info(f"Audio saved at: {audio_path}")

if __name__ == "__main__":
    main()
