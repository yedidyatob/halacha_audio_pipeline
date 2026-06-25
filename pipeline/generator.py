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

    def _get_generation_user_prompt(
        self, 
        gematria_siman: str, 
        master_context: str, 
        commentators_desc: str = 'השפתי כהן (הש"ך) והטורי זהב (הט"ז) והמקורות שהם מביאים'
    ) -> str:
        return (
            f"להלן קובץ המקורות עבור סימן {gematria_siman}:\n"
            f"{master_context}\n\n"
            f"אנא הפק את מבנה הנתונים ההיררכי (Data Map) עבור סימן {gematria_siman} על בסיס מקורות אלו, בהתאם מדויק להנחיות המערכת ולסכמה הנדרשת."
        )

    def _get_relations_user_prompt(self, drafts: dict) -> str:
        prompt_parts = [
            "להלן מבני הנתונים ההיררכיים של הסימנים השונים:",
            ""
        ]
        for siman, draft in sorted(drafts.items()):
            prompt_parts.append(f"=== סימן {siman} ===")
            prompt_parts.append(draft)
            prompt_parts.append("")
            
        prompt_parts.append(
            "אנא נתח כעת באופן מעמיק את כל הקשרים, התלויות, והעקרונות ההלכתיים המשותפים ביניהם לפי ההנחיות."
        )
        return "\n".join(prompt_parts)

    def _get_polishing_user_prompt(self, script_text: str, relations_text: str = "") -> str:
        if not relations_text:
            return (
                f"אנא ערוך כעת את רשימת הנתונים המבנית לשיעור שמע מלוטש לפי כל ההנחיות.\n\n"
                f"להלן מבנה הנתונים ההיררכי שחולץ בשלב 1 עבור השיעור:\n"
                f"{script_text}"
            )
            
        return (
            f"להלן מסמך הקשרים והזיקות ההלכתיות לסימנים השכנים:\n"
            f"{relations_text}\n\n"
            f"אנא ערוך כעת את רשימת הנתונים המבנית לשיעור שמע מלוטש לפי כל ההנחיות. "
            f"שלב את הקשרים והזיקות ההלכתיות לסימנים השכנים באופן טבעי וקולח בתוך השיעור (למשל, במעברים בין הסעיפים או בסיכומים).\n\n"
            f"להלן מבנה הנתונים ההיררכי שחולץ בשלב 1 עבור השיעור:\n"
            f"{script_text}"
        )

    @abstractmethod
    def generate_siman_script(
        self, 
        siman: int, 
        master_context: str, 
        system_instruction: str,
        commentators_desc: str = 'השפתי כהן (הש"ך) והטורי זהב (הט"ז) והמקורות שהם מביאים'
    ) -> str:
        """
        Generates a custom script for a specific Siman using the Master Reference Strategy.
        """
        pass

    @abstractmethod
    def analyze_cross_relations(self, drafts: dict, relations_instruction: str) -> str:
        """
        Analyzes thematic and conceptual connections across multiple Siman drafts.
        """
        pass

    @abstractmethod
    def polish_siman_script(self, script_text: str, relations_text: str, polishing_instruction: str) -> str:
        """
        Refines a draft script to ensure it is 100% optimized for TTS pronunciation.
        """
        pass


class BatchCapableGenerator(ABC):
    """
    Interface for generators that support asynchronous batch operations.
    """
    @abstractmethod
    def create_batch_generation_job(
        self, 
        siman: int, 
        master_context: str, 
        system_instruction: str,
        batch_input_path: str,
        commentators_desc: str = 'השפתי כהן (הש"ך) והטורי זהב (הט"ז) והמקורות שהם מביאים'
    ) -> str:
        pass

    @abstractmethod
    def retrieve_batch_result(self, batch_id: str) -> dict:
        pass


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
        system_instruction: str,
        commentators_desc: str = 'השפתי כהן (הש"ך) והטורי זהב (הט"ז) והמקורות שהם מביאים'
    ) -> str:
        gematria_siman = int_to_gematria(siman)
        logger.info(f"Generating script for Siman {siman} ({gematria_siman}) using Gemini model {self.model_name}...")
        user_prompt = self._get_generation_user_prompt(gematria_siman, master_context, commentators_desc)

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

    def analyze_cross_relations(self, drafts: dict, relations_instruction: str) -> str:
        logger.info(f"Analyzing cross-relations for {list(drafts.keys())} via Gemini model {self.model_name}...")
        user_prompt = self._get_relations_user_prompt(drafts)
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=relations_instruction,
                temperature=self.temperature
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config
            )
            
            relations_text = response.text
            if not relations_text:
                raise ValueError("Gemini API returned an empty text response for relations analysis.")
                
            logger.info("Successfully generated cross-relations analysis.")
            return relations_text
            
        except Exception as e:
            logger.error(f"Error during Gemini relations analysis: {e}")
            raise

    def polish_siman_script(self, script_text: str, relations_text: str, polishing_instruction: str) -> str:
        logger.info("Polishing generated script for TTS compliance via Gemini...")
        
        user_prompt = self._get_polishing_user_prompt(script_text, relations_text)
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=polishing_instruction,
                temperature=0.1
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
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


class OpenAIScriptGenerator(BaseScriptGenerator, BatchCapableGenerator):
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
        system_instruction: str,
        commentators_desc: str = 'השפתי כהן (הש"ך) והטורי זהב (הט"ז) והמקורות שהם מביאים'
    ) -> str:
        gematria_siman = int_to_gematria(siman)
        logger.info(f"Generating script for Siman {siman} ({gematria_siman}) using OpenAI model {self.model_name}...")
        user_prompt = self._get_generation_user_prompt(gematria_siman, master_context, commentators_desc)

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

    def analyze_cross_relations(self, drafts: dict, relations_instruction: str) -> str:
        logger.info(f"Analyzing cross-relations for {list(drafts.keys())} via OpenAI model {self.model_name}...")
        user_prompt = self._get_relations_user_prompt(drafts)
        
        messages = []
        is_reasoning_model = self.model_name.startswith("o1") or self.model_name.startswith("o3")
        
        if is_reasoning_model:
            messages.append({"role": "developer", "content": relations_instruction})
        else:
            messages.append({"role": "system", "content": relations_instruction})
            
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
            relations_text = response.choices[0].message.content
            if not relations_text:
                raise ValueError("OpenAI API returned an empty text response for relations analysis.")
                
            logger.info("Successfully generated cross-relations analysis.")
            return relations_text
            
        except Exception as e:
            logger.error(f"Error during OpenAI relations analysis: {e}")
            raise

    def polish_siman_script(self, script_text: str, relations_text: str, polishing_instruction: str) -> str:
        logger.info("Polishing generated script for TTS compliance via OpenAI...")
        
        user_prompt = self._get_polishing_user_prompt(script_text, relations_text)
        
        messages = []
        is_reasoning_model = self.model_name.startswith("o1") or self.model_name.startswith("o3")
        
        if is_reasoning_model:
            messages.append({"role": "developer", "content": polishing_instruction})
        else:
            messages.append({"role": "system", "content": polishing_instruction})
            
        messages.append({"role": "user", "content": user_prompt})
        
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
        batch_input_path: str,
        commentators_desc: str = 'השפתי כהן (הש"ך) והטורי זהב (הט"ז) והמקורות שהם מביאים'
    ) -> str:
        gematria_siman = int_to_gematria(siman)
        logger.info(f"Creating OpenAI Batch generation job specification for Siman {siman} ({gematria_siman})...")
        user_prompt = self._get_generation_user_prompt(gematria_siman, master_context, commentators_desc)

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
