import os
from abc import ABC, abstractmethod
from google import genai
from google.genai import types
from openai import OpenAI
from pipeline.gematria import int_to_gematria
from pipeline.logger import get_logger

logger = get_logger(__name__)

class BaseScriptGenerator(ABC):
    """
    Abstract Base Class for Halacha Script Generators.
    """
    def __init__(self, api_key: str = None, model_name: str = "", temperature: float = 0.3):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def generate_siman_script(
        self, 
        siman: int, 
        master_context: str, 
        system_instruction: str
    ) -> str:
        """
        Generates a custom script for a specific Siman using the Master Reference Strategy.
        """
        pass

    @abstractmethod
    def polish_siman_script(self, script_text: str) -> str:
        """
        Refines a draft script to ensure it is 100% optimized for TTS pronunciation.
        """
        pass

    def create_batch_generation_job(
        self, 
        siman: int, 
        master_context: str, 
        system_instruction: str,
        batch_input_path: str
    ) -> str:
        raise NotImplementedError("Batch processing is not implemented for this generator.")

    def retrieve_batch_result(self, batch_id: str) -> dict:
        raise NotImplementedError("Batch retrieval is not implemented for this generator.")


class GeminiScriptGenerator(BaseScriptGenerator):
    """
    Generates custom, TTS-optimized Hebrew script files using Gemini API (google-genai SDK).
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.5-flash", temperature: float = 0.3):
        super().__init__(api_key, model_name, temperature)
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
            
        try:
            self.client = genai.Client(**kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Client: {e}")
            raise

    def generate_siman_script(
        self, 
        siman: int, 
        master_context: str, 
        system_instruction: str
    ) -> str:
        gematria_siman = int_to_gematria(siman)
        logger.info(f"Generating script for Siman {siman} ({gematria_siman}) using Gemini model {self.model_name}...")

        user_prompt = (
            f"להלן קובץ המקורות המלא הכולל מספר סימנים של הטור, בית יוסף, שולחן ערוך, ש\"ך וט\"ז.\n"
            f"אנא התמקד כעת והפק שיעור שמע (אודיו) מפורט, שלם ומעמיק במיוחד עבור סימן {gematria_siman}.\n\n"
            f"דרישות קריטיות להשלמה (חובה ליישם):\n"
            f"1. עליך להקיף את כל הדעות, הסוגיות, והפרטים המופיעים בסימן {gematria_siman} עצמו. אל תסכם באופן כללי ואל תדלג על אף קושיה או שיטה.\n"
            f"2. מבנה פדגוגי הלכתי (Beit Midrash Flow): עבור כל דין או מחלוקת, התחל תמיד בהסבר המקור מן הטור והבית יוסף ודעות הראשונים השונות (בציון שמותיהם המפורשים), לאחר מכן הבא את פסק מרן המחבר בשולחן ערוך והגהת הרמ\"א, ולבסוף העמק בביאור דעות נושאי הכלים—השפתי כהן (הש\"ך) והטורי זהב (הט\"ז) והמקורות שהם מביאים.\n"
            f"3. ציין במפורש את שמות בעלי הדעות (הראשונים, האחרונים, והמפרשים) לאורך כל השיעור. הימנע לחלוטין מביטויים סתמיים כמו \"יש מי שסובר\".\n"
            f"4. הראה קשר רחב לסימנים השונים המופיעים בקובץ המקורות. אם יסוד או דעה פה מתקשרים לדיון בסימנים אחרים בקובץ, ציין זאת במפורש במילים שלך.\n"
            f"5. הקפד על התאמה מלאה ל-TTS: ללא סימני מרקדאון (בלי כוכביות, בלי כותרות, בלי רשימות), פתח את כל ראשי התיבות (למילים מלאות), ואל תשתמש בסוגריים כלל.\n\n"
            f"קובץ המקורות:\n"
            f"{master_context}"
        )

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=self.temperature
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config
            )
            
            script_text = response.text
            if not script_text:
                raise ValueError("Gemini API returned an empty text response.")
                
            logger.info(f"Successfully generated script for Siman {siman} ({len(script_text)} chars).")
            return script_text
            
        except Exception as e:
            logger.error(f"Error during Gemini script generation for Siman {siman}: {e}")
            raise

    def polish_siman_script(self, script_text: str) -> str:
        logger.info("Polishing generated script for TTS compliance via Gemini...")
        
        system_instruction = (
            "אתה עורך לשוני מקצועי ומומחה להכנת תמלילים עבור מערכות הקראת קול (Text-to-Speech) בעברית תורנית.\n"
            "תפקידך לקבל תמליל של שיעור הלכה בעברית, ולבצע הגהה ועריכה קפדנית כדי להבטיח שהוא מותאם לחלוטין לקריאה חלקה על ידי קריין ממוחשב (TTS), תוך שמירה על המבנה והתוכן המקורי במלואו.\n\n"
            "חוקי עריכה מחייבים (אל תסטה מהם!):\n"
            "1. הסר לחלוטין כל סימן עיצוב כגון כוכביות (*), כותרות (#), קווים תחתונים, הדגשות, או נקודות בולטים של רשימות. הטקסט חייב להיות פסקאות רציפות בלבד.\n"
            "2. פתח סעיפים וסימנים למילים מלאות בהגייה פונטית נכונה (TTS יקריין אותיות בודדות כמילים שגויות):\n"
            "   - \"סימן צד\" -> \"סימן צדיק דלת\"\n"
            "   - \"סימן צה\" -> \"סימן צדיק הא\"\n"
            "   - \"סימן צו\" -> \"סימן צדיק ו\"ו\"\n"
            "   - \"סימן צח\" -> \"סימן צדיק חית\"\n"
            "   - \"סימן קג\" -> \"סימן קוף גמל\"\n"
            "   - \"סעיף א\" -> \"סעיף אלף\"\n"
            "   - \"סעיף ב\" -> \"סעיף בית\"\n"
            "   - \"סעיף ג\" -> \"סעיף גימל\"\n"
            "   - \"סעיף ד\" -> \"סעיף דלת\"\n"
            "   - \"סעיף ה\" -> \"סעיף הא\"\n"
            "   - \"סעיף ו\" -> \"סעיף ו\"ו\"\n"
            "   - \"סעיף ז\" -> \"סעיף זין\"\n"
            "   וכן הלאה לכל מספר או אות הלכתית.\n"
            "3. פתח ראשי תיבות נפוצים להגייה שלמה:\n"
            "   - \"חנ\"ן\" -> \"חתיכה נעשית נבילה\"\n"
            "   - \"נ\"ט בר נ\"ט\" -> \"נותן טעם בר נותן טעם\"\n"
            "   - \"שו\"ע\" -> \"שולחן ערוך\"\n"
            "   - \"ב\"ח\" -> \"בשר בחלב\"\n"
            "   - \"יו\"ד\" -> \"יורה דעה\"\n"
            "   - \"רמ\"א\" -> \"הרמ\"א\"\n"
            "   - \"ש\"ך\" -> \"הש\"ך\" (או \"השפתי כהן\")\n"
            "   - \"ט\"ז\" -> \"הט\"ז\" (או \"הטורי זהב\")\n"
            "   - \"מהרר\"י\" -> \"רבי ישראל\"\n"
            "   - \"מהרש\"ל\" -> \"המהרש\"ל\"\n"
            "4. הימנע לחלוטין מסוגריים ( ). אם יש טקסט בסוגריים, שלב אותו בתוך זרימת המשפט או מחק את הסוגריים והשאר רק את הטקסט הפנימי.\n"
            "5. שמור על לשון ישיבתית-תורנית נקייה, והסר ביטויים חילוניים או מודרניים מדי שמשתחלים לעיתים (כמו \"פתחו את הלבבות והראש\").\n"
            "6. פלט אך ורק את הטקסט המוגה והערוך, ללא הקדמות עצמיות (כמו \"הנה הטקסט המתוקן\") וללא הערות שוליים."
        )
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=script_text,
                config=config
            )
            
            polished_text = response.text
            if not polished_text:
                raise ValueError("Gemini API returned an empty text response for polish call.")
                
            logger.info(f"Successfully polished script ({len(polished_text)} chars).")
            return polished_text
            
        except Exception as e:
            logger.error(f"Error during Gemini script polish call: {e}")
            raise


class OpenAIScriptGenerator(BaseScriptGenerator):
    """
    Generates custom, TTS-optimized Hebrew script files using OpenAI's API.
    Supports o1/o3 reasoning models automatically.
    """
    def __init__(self, api_key: str = None, model_name: str = "o1", temperature: float = 1.0, service_tier: str = None):
        super().__init__(api_key, model_name, temperature)
        self.service_tier = service_tier
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
            
        try:
            self.client = OpenAI(**kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI Client: {e}")
            raise

    def generate_siman_script(
        self, 
        siman: int, 
        master_context: str, 
        system_instruction: str
    ) -> str:
        gematria_siman = int_to_gematria(siman)
        logger.info(f"Generating script for Siman {siman} ({gematria_siman}) using OpenAI model {self.model_name}...")

        user_prompt = (
            f"להלן קובץ המקורות המלא הכולל מספר סימנים של הטור, בית יוסף, שולחך ערוך, ש\"ך וט\"ז.\n"
            f"אנא התמקד כעת והפק שיעור שמע (אודיו) מפורט, שלם ומעמיק במיוחד עבור סימן {gematria_siman}.\n\n"
            f"דרישות קריטיות להשלמה (חובה ליישם):\n"
            f"1. עליך להקיף את כל הדעות, הסוגיות, והפרטים המופיעים בסימן {gematria_siman} עצמו. אל תסכם באופן כללי ואל תדלג על אף קושיה או שיטה.\n"
            f"2. מבנה פדגוגי הלכתי (Beit Midrash Flow): עבור כל דין או מחלוקת, התחל תמיד בהסבר המקור מן הטור והבית יוסף ודעות הראשונים השונות (בציון שמותיהם המפורשים), לאחר מכן הבא את פסק מרן המחבר בשולחן ערוך והגהת הרמ\"א, ולבסוף העמק בביאור דעות נושאי הכלים—השפתי כהן (הש\"ך) והטורי זהב (הט\"ז) והמקורות שהם מביאים.\n"
            f"3. ציין במפורש את שמות בעלי הדעות (הראשונים, האחרונים, והמפרשים) לאורך כל השיעור. הימנע לחלוטין מביטויים סתמיים כמו \"יש מי שסובר\".\n"
            f"4. הראה קשר רחב לסימנים השונים המופיעים בקובץ המקורות. אם יסוד או דעה פה מתקשרים לדיון בסימנים אחרים בקובץ, ציין זאת במפורש במילים שלך.\n"
            f"5. הקפד על התאמה מלאה ל-TTS: ללא סימני מרקדאון (בלי כוכביות, בלי כותרות, בלי רשימות), פתח את כל ראשי התיבות (למילים מלאות), ואל תשתמש בסוגריים כלל.\n\n"
            f"קובץ המקורות:\n"
            f"{master_context}"
        )

        messages = []
        is_reasoning_model = self.model_name.startswith("o1") or self.model_name.startswith("o3")
        
        if is_reasoning_model:
            messages.append({"role": "developer", "content": system_instruction})
        else:
            messages.append({"role": "system", "content": system_instruction})
            
        messages.append({"role": "user", "content": user_prompt})

        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
            }
            if not is_reasoning_model:
                kwargs["temperature"] = self.temperature
            if self.service_tier:
                kwargs["service_tier"] = self.service_tier

            response = self.client.chat.completions.create(**kwargs)
            script_text = response.choices[0].message.content
            if not script_text:
                raise ValueError("OpenAI API returned an empty text response.")
                
            logger.info(f"Successfully generated script for Siman {siman} ({len(script_text)} chars).")
            return script_text
            
        except Exception as e:
            logger.error(f"Error during OpenAI script generation for Siman {siman}: {e}")
            raise

    def polish_siman_script(self, script_text: str) -> str:
        logger.info("Polishing generated script for TTS compliance via OpenAI...")
        
        system_instruction = (
            "אתה עורך לשוני מקצועי ומומחה להכנת תמלילים עבור מערכות הקראת קול (Text-to-Speech) בעברית תורנית.\n"
            "תפקידך לקבל תמליל של שיעור הלכה בעברית, ולבצע הגהה ועריכה קפדנית כדי להבטיח שהוא מותאם לחלוטין לקריאה חלקה על ידי קריין ממוחשב (TTS), תוך שמירה על המבנה והתוכן המקורי במלואו.\n\n"
            "חוקי עריכה מחייבים (אל תסטה מהם!):\n"
            "1. הסר לחלוטין כל סימן עיצוב כגון כוכביות (*), כותרות (#), קווים תחתונים, הדגשות, או נקודות בולטים של רשימות. הטקסט חייב להיות פסקאות רציפות בלבד.\n"
            "2. פתח סעיפים וסימנים למילים מלאות בהגייה פונטית נכונה (TTS יקריין אותיות בודדות כמילים שגויות):\n"
            "   - \"סימן צד\" -> \"סימן צדיק דלת\"\n"
            "   - \"סימן צה\" -> \"סימן צדיק הא\"\n"
            "   - \"סימן צו\" -> \"סימן צדיק ו\"ו\"\n"
            "   - \"סימן צח\" -> \"סימן צדיק חית\"\n"
            "   - \"סימן קג\" -> \"סימן קוף גמל\"\n"
            "   - \"סעיף א\" -> \"סעיף אלף\"\n"
            "   - \"סעיף ב\" -> \"סעיף בית\"\n"
            "   - \"סעיף ג\" -> \"סעיף גימל\"\n"
            "   - \"סעיף ד\" -> \"סעיף דלת\"\n"
            "   - \"סעיף ה\" -> \"סעיף הא\"\n"
            "   - \"סעיף ו\" -> \"סעיף ו\"ו\"\n"
            "   - \"סעיף ז\" -> \"סעיף זין\"\n"
            "   וכן הלאה לכל מספר או אות הלכתית.\n"
            "3. פתח ראשי תיבות נפוצים להגייה שלמה:\n"
            "   - \"חנ\"ן\" -> \"חתיכה נעשית נבילה\"\n"
            "   - \"נ\"ט בר נ\"ט\" -> \"נותן טעם בר נותן טעם\"\n"
            "   - \"שו\"ע\" -> \"שולחן ערוך\"\n"
            "   - \"ב\"ח\" -> \"בשר בחלב\"\n"
            "   - \"יו\"ד\" -> \"יורה דעה\"\n"
            "   - \"רמ\"א\" -> \"הרמ\"א\"\n"
            "   - \"ש\"ך\" -> \"הש\"ך\" (או \"השפתי כהן\")\n"
            "   - \"ט\"ז\" -> \"הט\"ז\" (או \"הטורי זהב\")\n"
            "   - \"מהרר\"י\" -> \"רבי ישראל\"\n"
            "   - \"מהרש\"ל\" -> \"המהרש\"ל\"\n"
            "4. הימנע לחלוטין מסוגריים ( ). אם יש טקסט בסוגריים, שלב אותו בתוך זרימת המשפט או מחק את הסוגריים והשאר רק את הטקסט הפנימי.\n"
            "5. שמור על לשון ישיבתית-תורנית נקייה, והסר ביטויים חילוניים או מודרניים מדי שמשתחלים לעיתים (כמו \"פתחו את הלבבות והראש\").\n"
            "6. פלט אך ורק את הטקסט המוגה והערוך, ללא הקדמות עצמיות (כמו \"הנה הטקסט המתוקן\") וללא הערות שוליים."
        )
        
        messages = []
        is_reasoning_model = self.model_name.startswith("o1") or self.model_name.startswith("o3")
        
        if is_reasoning_model:
            messages.append({"role": "developer", "content": system_instruction})
        else:
            messages.append({"role": "system", "content": system_instruction})
            
        messages.append({"role": "user", "content": script_text})
        
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
            }
            if not is_reasoning_model:
                kwargs["temperature"] = 0.1
            if self.service_tier:
                kwargs["service_tier"] = self.service_tier

            response = self.client.chat.completions.create(**kwargs)
            polished_text = response.choices[0].message.content
            if not polished_text:
                raise ValueError("OpenAI API returned an empty text response for polish call.")
                
            logger.info(f"Successfully polished script ({len(polished_text)} chars).")
            return polished_text
            
        except Exception as e:
            logger.error(f"Error during OpenAI script polish call: {e}")
            raise

    def create_batch_generation_job(
        self, 
        siman: int, 
        master_context: str, 
        system_instruction: str,
        batch_input_path: str
    ) -> str:
        gematria_siman = int_to_gematria(siman)
        logger.info(f"Creating OpenAI Batch generation job specification for Siman {siman} ({gematria_siman})...")

        user_prompt = (
            f"להלן קובץ המקורות המלא הכולל מספר סימנים של הטור, בית יוסף, שולחך ערוך, ש\"ך וט\"ז.\n"
            f"אנא התמקד כעת והפק שיעור שמע (אודיו) מפורט, שלם ומעמיק במיוחד עבור סימן {gematria_siman}.\n\n"
            f"דרישות קריטיות להשלמה (חובה ליישם):\n"
            f"1. עליך להקיף את כל הדעות, הסוגיות, והפרטים המופיעים בסימן {gematria_siman} עצמו. אל תסכם באופן כללי ואל תדלג על אף קושיה או שיטה.\n"
            f"2. מבנה פדגוגי הלכתי (Beit Midrash Flow): עבור כל דין או מחלוקת, התחל תמיד בהסבר המקור מן הטור והבית יוסף ודעות הראשונים השונות (בציון שמותיהם המפורשים), לאחר מכן הבא את פסק מרן המחבר בשולחן ערוך והגהת הרמ\"א, ולבסוף העמק בביאור דעות נושאי הכלים—השפתי כהן (הש\"ך) והטורי זהב (הט\"ז) והמקורות שהם מביאים.\n"
            f"3. ציין במפורש את שמות בעלי הדעות (הראשונים, האחרונים, והמפרשים) לאורך כל השיעור. הימנע לחלוטין מביטויים סתמיים כמו \"יש מי שסובר\".\n"
            f"4. הראה קשר רחב לסימנים השונים המופיעים בקובץ המקורות. אם יסוד או דעה פה מתקשרים לדיון בסימנים אחרים בקובץ, ציין זאת במפורש במילים שלך.\n"
            f"5. הקפד על התאמה מלאה ל-TTS: ללא סימני מרקדאון (בלי כוכביות, בלי כותרות, בלי רשימות), פתח את כל ראשי התיבות (למילים מלאות), ואל תשתמש בסוגריים כלל.\n\n"
            f"קובץ המקורות:\n"
            f"{master_context}"
        )

        messages = []
        is_reasoning_model = self.model_name.startswith("o1") or self.model_name.startswith("o3")
        
        if is_reasoning_model:
            messages.append({"role": "developer", "content": system_instruction})
        else:
            messages.append({"role": "system", "content": system_instruction})
            
        messages.append({"role": "user", "content": user_prompt})

        import json
        body = {
            "model": self.model_name,
            "messages": messages,
        }
        if not is_reasoning_model:
            body["temperature"] = self.temperature

        batch_request = {
            "custom_id": f"siman_{siman}_generation",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }

        os.makedirs(os.path.dirname(batch_input_path), exist_ok=True)
        with open(batch_input_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
            
        logger.info(f"Uploading batch file {batch_input_path} to OpenAI...")
        with open(batch_input_path, "rb") as f:
            file_response = self.client.files.create(file=f, purpose="batch")
            
        logger.info(f"Starting batch job using file ID: {file_response.id}...")
        batch_response = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch_response.id

    def retrieve_batch_result(self, batch_id: str) -> dict:
        logger.info(f"Retrieving batch status for {batch_id}...")
        batch = self.client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            return {"status": batch.status}
            
        if not batch.output_file_id:
            logger.error(f"Batch completed but no output_file_id found: {batch}")
            return {"status": "failed", "error": "No output file generated by OpenAI."}
            
        logger.info(f"Downloading batch results from file ID: {batch.output_file_id}...")
        file_content = self.client.files.content(batch.output_file_id).text
        
        import json
        for line in file_content.strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            if "response" in data and "body" in data["response"] and "choices" in data["response"]["body"]:
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                return {
                    "status": "completed",
                    "content": content
                }
                
        return {"status": "failed", "error": "Target content choice not found in batch output lines."}
