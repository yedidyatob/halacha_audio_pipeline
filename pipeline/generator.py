import json
import os
from abc import ABC, abstractmethod
from google import genai
from google.genai import types
from openai import OpenAI
from pipeline.gematria import int_to_gematria
from pipeline.logger import get_logger
from pipeline.utils import log_gemini_usage

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

    def generate_text(self, system_instruction: str, user_prompt: str, thinking_level: str = None) -> str:
        """
        Generic single-shot completion (system instruction + user prompt) at this
        generator's configured temperature. Used by auxiliary stages such as
        the pronunciation stage. thinking_level is honoured by Gemini only (None = model default).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement generate_text().")


class BatchCapableGenerator(ABC):
    """
    Interface for generators that support asynchronous batch operations.
    Provides a unified 3-method interface for submitting, polling, and retrieving batch jobs.
    Works with both Gemini (inline requests) and OpenAI (JSONL file upload) batch APIs.
    """
    @abstractmethod
    def submit_batch(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float,
        custom_id: str,
        cache_dir: str = "./cache",
        thinking_level: str = None
    ) -> str:
        """
        Submit a single request as a batch job. Returns a job identifier string.
        thinking_level is honoured by Gemini only (None = model default).
        """
        pass

    @abstractmethod
    def get_batch_status(self, job_id: str) -> str:
        """
        Check the status of a batch job.
        Returns one of: 'pending', 'completed', 'failed'.
        """
        pass

    @abstractmethod
    def get_batch_result(self, job_id: str) -> str:
        """Retrieve the text content from a completed batch job."""
        pass


class GeminiScriptGenerator(BaseScriptGenerator, BatchCapableGenerator):
    """
    Generates custom, TTS-optimized Hebrew script files using Gemini API (google-genai SDK).
    """
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.5-flash", temperature: float = 0.3, ssl_verify: bool = True):
        super().__init__(api_key, model_name, temperature)
        self.ssl_verify = ssl_verify
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        
        # Configure httpx client with SSL verification setting
        if not ssl_verify:
            try:
                import urllib3
                # Disable SSL warnings when verification is disabled
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                # Pass verify=False to the underlying httpx client via http_options
                kwargs["http_options"] = {"client_args": {"verify": False}}
            except ImportError:
                logger.warning("urllib3 not available for disabling SSL warnings")
            
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
            
            log_gemini_usage(logger, response, self.model_name)
            
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
            
            log_gemini_usage(logger, response, self.model_name)
            
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
            
            log_gemini_usage(logger, response, self.model_name)
            
            polished_text = response.text
            if not polished_text:
                raise ValueError("Gemini API returned an empty text response for polish call.")
                
            logger.info(f"Successfully polished script ({len(polished_text)} chars).")
            return polished_text
            
        except Exception as e:
            logger.error(f"Error during Gemini script polish call: {e}")
            raise

    def generate_text(self, system_instruction: str, user_prompt: str, thinking_level: str = None) -> str:
        logger.info(
            f"Running Gemini text call via model {self.model_name} (temperature {self.temperature}"
            + (f", thinking_level {thinking_level}" if thinking_level else "") + ")..."
        )
        config_kwargs = {"system_instruction": system_instruction, "temperature": self.temperature}
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        config = types.GenerateContentConfig(**config_kwargs)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=config
        )
        log_gemini_usage(logger, response, self.model_name)
        text = response.text
        if not text:
            raise ValueError("Gemini API returned an empty text response.")
        return text

    def submit_batch(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float,
        custom_id: str,
        cache_dir: str = "./cache",
        thinking_level: str = None
    ) -> str:
        logger.info(f"Submitting Gemini batch job (custom_id: {custom_id})...")
        request_config = {
            'system_instruction': system_instruction,
            'temperature': temperature
        }
        if thinking_level:
            request_config['thinking_config'] = {'thinking_level': thinking_level}
        inline_request = {
            'contents': [{'role': 'user', 'parts': [{'text': user_prompt}]}],
            'config': request_config
        }
        batch_job = self.client.batches.create(
            model=self.model_name,
            src=[inline_request]
        )
        logger.info(f"Gemini batch job created: {batch_job.name} (custom_id: {custom_id})")
        return batch_job.name

    def get_batch_status(self, job_id: str) -> str:
        job = self.client.batches.get(name=job_id)
        state_str = str(job.state)
        if "SUCCEEDED" in state_str:
            return "completed"
        elif "FAILED" in state_str or "CANCELLED" in state_str:
            return "failed"
        return "pending"

    def get_batch_result(self, job_id: str) -> str:
        """
        Text of the job's single inlined response. A job can SUCCEED while its request failed:
        the InlinedResponse then carries `error` (code / message) instead of `response`.
        Raises ValueError with the API's reason for any failed, blocked or empty result.
        """
        job = self.client.batches.get(name=job_id)
        responses = (job.dest.inlined_responses if job.dest else None) or []
        if not responses:
            raise ValueError(f"Gemini batch job {job_id} has no inlined response.")
        inlined = responses[0]
        if inlined.error is not None:
            raise ValueError(f"Gemini batch request failed (job {job_id}): {_format_job_error(inlined.error)}")
        if inlined.response is None:
            raise ValueError(f"Gemini batch job {job_id} returned neither a response nor an error.")
        return _gemini_response_text(inlined.response, f"Gemini batch job {job_id}")


# google.rpc.Code names, for the code of a failed batch request (JobError.code)
_RPC_CODE_NAMES = {
    0: "OK", 1: "CANCELLED", 2: "UNKNOWN", 3: "INVALID_ARGUMENT", 4: "DEADLINE_EXCEEDED", 5: "NOT_FOUND",
    6: "ALREADY_EXISTS", 7: "PERMISSION_DENIED", 8: "RESOURCE_EXHAUSTED", 9: "FAILED_PRECONDITION",
    10: "ABORTED", 11: "OUT_OF_RANGE", 12: "UNIMPLEMENTED", 13: "INTERNAL", 14: "UNAVAILABLE",
    15: "DATA_LOSS", 16: "UNAUTHENTICATED",
}


def _format_job_error(error: types.JobError) -> str:
    """e.g. 'error code 1 (CANCELLED): The operation was cancelled.'"""
    name = f" ({_RPC_CODE_NAMES[error.code]})" if error.code in _RPC_CODE_NAMES else ""
    details = f" (details: {'; '.join(error.details)})" if error.details else ""
    return f"error code {error.code}{name}: {error.message or 'no message'}{details}"


def _enum_name(value):
    """'MAX_TOKENS' rather than 'FinishReason.MAX_TOKENS' (an unknown value from the API stays a plain string)."""
    return getattr(value, "name", value)


def _gemini_response_text(response: types.GenerateContentResponse, source: str) -> str:
    """The answer text (thought parts excluded); ValueError naming the block / finish reason otherwise."""
    feedback = response.prompt_feedback
    if feedback and feedback.block_reason:
        message = f": {feedback.block_reason_message}" if feedback.block_reason_message else ""
        raise ValueError(f"{source}: prompt blocked ({_enum_name(feedback.block_reason)}){message}")
    if not response.candidates:
        raise ValueError(f"{source}: response has no candidates.")
    candidate = response.candidates[0]
    finish = _enum_name(candidate.finish_reason)
    parts = (candidate.content.parts if candidate.content else None) or []
    text = "".join(part.text for part in parts if part.text and not part.thought)
    if finish not in (None, types.FinishReason.STOP.name, types.FinishReason.FINISH_REASON_UNSPECIFIED.name):
        message = f": {candidate.finish_message}" if candidate.finish_message else ""
        raise ValueError(f"{source}: generation stopped early (finish_reason {finish}){message}"
                         + (f"; {len(text)} chars of partial text discarded" if text else ""))
    if not text:
        raise ValueError(f"{source}: response has no text (finish_reason {finish}).")
    return text


class OpenAIScriptGenerator(BaseScriptGenerator, BatchCapableGenerator):
    """
    Generates custom, TTS-optimized Hebrew script files using OpenAI's API.
    Supports o1/o3 reasoning models automatically.
    """
    def __init__(self, api_key: str = None, model_name: str = "o1", temperature: float = 1.0, service_tier: str = None, ssl_verify: bool = True):
        super().__init__(api_key, model_name, temperature)
        self.service_tier = service_tier
        self.ssl_verify = ssl_verify
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        
        # Configure httpx client with SSL verification setting
        if not ssl_verify:
            try:
                import httpx
                import urllib3
                # Disable SSL warnings when verification is disabled
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                kwargs["http_client"] = httpx.Client(verify=False, timeout=300.0)
            except ImportError:
                logger.warning("httpx not available, SSL verification setting may not be applied")
            
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

    def generate_text(self, system_instruction: str, user_prompt: str, thinking_level: str = None) -> str:
        # thinking_level is Gemini-only; ignored here.
        logger.info(f"Running OpenAI text call via model {self.model_name}...")
        is_reasoning_model = self.model_name.startswith("o1") or self.model_name.startswith("o3")
        messages = [
            {"role": "developer" if is_reasoning_model else "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {"model": self.model_name, "messages": messages}
        if not is_reasoning_model:
            kwargs["temperature"] = self.temperature
        if self.service_tier:
            kwargs["service_tier"] = self.service_tier
        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content
        if not text:
            raise ValueError("OpenAI API returned an empty text response.")
        return text

    def submit_batch(
        self,
        system_instruction: str,
        user_prompt: str,
        temperature: float,
        custom_id: str,
        cache_dir: str = "./cache",
        thinking_level: str = None
    ) -> str:
        # thinking_level is Gemini-only; ignored here.
        logger.info(f"Submitting OpenAI batch job (custom_id: {custom_id})...")

        messages = []
        is_reasoning_model = self.model_name.startswith("o1") or self.model_name.startswith("o3")

        if is_reasoning_model:
            messages.append({"role": "developer", "content": system_instruction})
        else:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({"role": "user", "content": user_prompt})

        body = {
            "model": self.model_name,
            "messages": messages,
        }
        if not is_reasoning_model:
            body["temperature"] = temperature

        batch_request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }

        batch_input_path = os.path.join(cache_dir, f"openai_batch_{custom_id}.jsonl")
        os.makedirs(cache_dir, exist_ok=True)
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
        logger.info(f"OpenAI batch job created: {batch_response.id} (custom_id: {custom_id})")
        return batch_response.id

    def get_batch_status(self, job_id: str) -> str:
        batch = self.client.batches.retrieve(job_id)
        if batch.status == "completed":
            return "completed"
        elif batch.status in ("failed", "cancelled", "expired"):
            return "failed"
        return "pending"

    def get_batch_result(self, job_id: str) -> str:
        """
        Text of the batch's single request. A completed batch can hold a failed request: it is
        then in the error file (or carries `error` / a non-200 response). Raises ValueError with
        the API's reason for any failed, cut-off or empty result.
        """
        logger.info(f"Retrieving OpenAI batch result for {job_id}...")
        batch = self.client.batches.retrieve(job_id)
        records = []
        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id:
                content = self.client.files.content(file_id).text
                records.extend(json.loads(line) for line in content.splitlines() if line.strip())
        if not records:
            raise ValueError(f"OpenAI batch {job_id} completed without an output or error file.")
        record = records[0]
        source = f"OpenAI batch {job_id}"
        error = record.get("error")
        if error:
            raise ValueError(f"{source}: request failed (error code {error.get('code')}: {error.get('message')})")
        response = record.get("response") or {}
        body = response.get("body") or {}
        if response.get("status_code") != 200:
            message = (body.get("error") or {}).get("message", "no message")
            raise ValueError(f"{source}: request failed (HTTP {response.get('status_code')}: {message})")
        choices = body.get("choices") or []
        if not choices:
            raise ValueError(f"{source}: response has no choices.")
        finish = choices[0].get("finish_reason")
        text = (choices[0].get("message") or {}).get("content")
        if finish not in (None, "stop"):
            raise ValueError(f"{source}: generation stopped early (finish_reason {finish})")
        if not text:
            raise ValueError(f"{source}: response has no text (finish_reason {finish}).")
        return text
