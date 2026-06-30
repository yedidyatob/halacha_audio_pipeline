import os
import sys
import logging

# Add the pipeline module to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.tts import GeminiTTS
from pipeline.config import PipelineConfig
from mutagen.mp3 import MP3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTS_Test")

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    config = PipelineConfig(config_path)

    # Hardcode the single paragraph test text (433 characters)
    test_text = (
        "מכאן אנו עוברים לסעיף זין, ועוברים מנפילה פסיבית לפעולה אקטיבית של חיתוך. המחבר פוסק "
        "שבשר רותח שחתכו בסכין חלבית, כל החתיכה אסורה אם אין בה שישים כנגד מקום הסכין שחתך. "
        "אבל אם הסכין אינו בן יומו, אינו אוסר אלא כדי קליפה. הרמ\"א מוסיף שכל זה בבשר רותח בכלי ראשון, "
        "ואז אם הסכין בן יומו ואין שישים, הכל אסור ואף הסכין צריך הגעלה. אבל אם הוא כלי שני, הבשר "
        "צריך קליפה והסכין נעיצה בקרקע. ואפילו אם הסכין אינו בן יומו, יש לקלוף את הבשר מעט משום "
        "שמנונית הסכין."
    )
    
    logger.info(f"Test Text Length: {len(test_text)} characters")

    # Load API keys
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=r"c:\Users\yedidyat\.gemini\antigravity\scratch\halacha_audio_pipeline\.env", override=True)
    if not config.gemini_api_key:
        config.gemini_api_key = os.environ.get("GEMINI_API_KEY")

    # Initialize Gemini TTS (this uses gemini-3.1-flash-tts-preview if configured)
    tts_engine = GeminiTTS(
        model_id=config.gemini_tts_model,
        voice_name=config.gemini_tts_voice,
        temperature=config.gemini_tts_temp,
        api_key=config.gemini_api_key
    )

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'test_audio')
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "saif_zayin_para1.mp3")
    
    # We will synthesize the speech and rely on the internal logging we added to `utils.py`
    # to output the tokens and exact dollar cost to the terminal.
    logger.info("Sending TTS request to Gemini API...")
    tts_engine.synthesize(test_text, audio_path)
    
    if os.path.exists(audio_path):
        audio = MP3(audio_path)
        duration_sec = audio.info.length
        logger.info(f"Generated Audio Duration: {duration_sec:.2f} seconds")
        logger.info(f"Done. Audio saved at: {audio_path}")
    else:
        logger.error("Audio file was not generated.")

if __name__ == "__main__":
    main()
