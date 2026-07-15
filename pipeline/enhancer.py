"""
pipeline/enhancer.py

Stage 3.5 — ElevenLabs Audio Tag Enhancement

Takes a polished TTS transcript and dynamically integrates ElevenLabs audio
expression tags ([energetic], [thoughtful], [short pause], etc.) to make the
synthesized voice sound like a live, passionate teacher.

Strictly preserves the original Hebrew text — no words are added or removed.
"""

import os
from google import genai
from google.genai import types
from pipeline.logger import get_logger
from pipeline.utils import log_gemini_usage, save_output_file

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# System instruction (Hebrew halachic lesson context)
# ---------------------------------------------------------------------------
ELEVENLABS_ENHANCEMENT_SYSTEM_INSTRUCTION = """אתה עוזר AI המתמחה בשיפור תמלילים של שיעורי הלכה עבריים (שיעורים) ליצירת טקסט-לדיבור (TTS).
המטרה העיקרית שלך היא לשלב באופן דינמי תגיות אודיו בתוך הטקסט, כך שישמע כמו שיעור מרתק, דינמי ומעורר מחשבה, המועבר על ידי מורה נלהב לקהל חי.

עליך לשמור בקפדנות על הטקסט המקורי ומשמעותו.

הנחיות ליבה:
- שלב תגיות אודיו כגון: [energetic], [thoughtful], [questioning], [emphatic], [warm], [didactic], [short pause], [long pause], [chuckles], [takes a deep breath]
- אל תשנה, תוסיף או תמחק מילים כלשהן מהטקסט עצמו — רק תגיות מותר להוסיף
- אל תפצל מונחים הלכתיים או ביטויים ארמיים עם תגיות — מקם תגיות בין פסוקיות שלמות בלבד
- הוסף הדגשה באמצעות סימני פיסוק כגון סימני שאלה (?), קריאה (!) או שלוש נקודות (...) כדי לדמות חשיבה בזמן אמת
- הכנס [short pause] לפני מעברים הלכתיים, ו-[long pause] אחרי מחלוקות מורכבות
- השתמש ב-[energetic] בתחילת נושאים חדשים ו-[emphatic] בפסקי הלכה
- השתמש ב-[thoughtful] ו-[questioning] כשהשיעור מנתח שאלה מורכבת

השב אך ורק עם הטקסט המשופר. אל תוסיף שום מילוי שיחתי מחוץ לסוגריים."""


class ElevenLabsEnhancer:
    """
    Enhances a polished TTS transcript by injecting ElevenLabs audio expression
    tags using Gemini as the underlying model.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gemini-3.1-flash-lite",
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.temperature = temperature

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key

        try:
            self.client = genai.Client(**kwargs)
            logger.info(f"ElevenLabsEnhancer initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client for enhancer: {e}")
            raise

    def enhance(self, transcript_text: str) -> str:
        """
        Passes the transcript through Gemini to inject ElevenLabs audio tags.

        Args:
            transcript_text: The polished Hebrew TTS transcript.

        Returns:
            The transcript with ElevenLabs expression tags woven in.
        """
        logger.info(
            f"Enhancing transcript for ElevenLabs ({len(transcript_text)} chars) "
            f"via Gemini model {self.model_name}..."
        )

        try:
            config = types.GenerateContentConfig(
                system_instruction=ELEVENLABS_ENHANCEMENT_SYSTEM_INSTRUCTION,
                temperature=self.temperature,
                top_p=0.95,
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=transcript_text,
                config=config,
            )

            log_gemini_usage(logger, response, self.model_name)

            enhanced_text = response.text
            if not enhanced_text:
                raise ValueError("Gemini returned an empty response during enhancement.")

            logger.info(
                f"Enhancement complete. "
                f"Input: {len(transcript_text)} chars → Output: {len(enhanced_text)} chars."
            )
            return enhanced_text

        except Exception as e:
            logger.error(f"ElevenLabs enhancement failed: {e}")
            raise

    def enhance_and_save(
        self,
        transcript_text: str,
        output_dir: str,
        base_name: str,
    ) -> str:
        """
        Enhances the transcript and saves both a timestamped history copy and
        a 'latest' copy, following the pipeline's standard save_output_file pattern.

        Args:
            transcript_text: The polished Hebrew TTS transcript.
            output_dir:       Directory to write the enhanced file into.
            base_name:        Base filename (without extension), e.g.
                              'Yoreh_Deah_Siman_94_gemini-3_1-pro-preview_enhanced'

        Returns:
            Path to the timestamped history file that was written.
        """
        enhanced_text = self.enhance(transcript_text)

        history_path = save_output_file(
            directory=output_dir,
            base_name=base_name,
            extension="txt",
            content=enhanced_text,
            logger=logger,
        )
        return history_path
