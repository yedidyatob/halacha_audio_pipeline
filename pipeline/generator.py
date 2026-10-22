import json
import os
from abc import ABC, abstractmethod
from google import genai
from google.genai import types
from openai import OpenAI
from pipeline.gematria import int_to_gematria
from pipeline.logger import get_logger
from typing import Optional
from pipeline.utils import log_gemini_usage, save_output_file

logger = get_logger(__name__)


class BatchResultError(ValueError):
    """
    A finished batch job has no usable result: its request failed, or the response is blocked,
    cut off or empty. Fetching again will not help; a new batch job may. (A network / API error
    while fetching is not a BatchResultError: the result may still be there.)
    """

class BaseScriptGenerator(ABC):
    """
    Abstract Base Class for Halacha Script Generators.
    """
    def __init__(self, api_key: Optional[str], model_name: str, temperature: float):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature

    def _get_generation_user_prompt(
        self, 
        gematria_siman: str, 
        master_context: str,
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
        thinking_level: Optional[str] = None,
        thinking_name: Optional[str] = None,
    ) -> str:
        """
        Generates a custom script for a specific Siman using the Master Reference Strategy.
        """
        pass

    @abstractmethod
    def analyze_cross_relations(self, drafts: dict, relations_instruction: str,
                                thinking_name: Optional[str] = None) -> str:
        """
        Analyzes thematic and conceptual connections across multiple Siman drafts.
        """
        pass

    @abstractmethod
    def polish_siman_script(self, script_text: str, relations_text: str, polishing_instruction: str,
                            temperature: float,
                            thinking_name: Optional[str] = None) -> str:
        """
        Refines a draft script to ensure it is 100% optimized for TTS pronunciation.
        """
        pass

    def generate_text(self, system_instruction: str, user_prompt: str, thinking_level: str = None,
                      thinking_name: Optional[str] = None) -> str:
        """
        Generic single-shot completion (system instruction + user prompt) at this
        generator's configured temperature. Used by auxiliary stages such as
        the transcript enhancer. thinking_level is honoured by Gemini only (None = model default).
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
        """
        The text of a completed batch job. Raises BatchResultError if the job has no usable
        result; any other exception means the fetch itself failed (the result may still be there).
        """
        pass


class GeminiScriptGenerator(BaseScriptGenerator, BatchCapableGenerator):
    """
    Generates custom, TTS-optimized Hebrew script files using Gemini API (google-genai SDK).

    Every synchronous call streams (generate_content_stream) with thought summaries enabled, so
    bytes keep flowing while the model thinks: a long non-streaming call sends nothing until it
    finishes, and an idle connection can be closed by a proxy or front end ("Server disconnected
    without sending a response"). Only the answer text is returned; the thought summaries are
    written to <thinking_dir>/<thinking_name>_thinking.txt when the caller names the call.
    """
    def __init__(self, api_key: str, model_name: str, temperature: float, ssl_verify: bool,
                 pricing: dict, thinking_dir: str):
        super().__init__(api_key, model_name, temperature)
        self.ssl_verify = ssl_verify
        self.pricing = pricing
        self.thinking_dir = thinking_dir
        self.client = build_gemini_client(api_key, ssl_verify)

    def _generate(self, system_instruction: str, user_prompt: str, temperature: float,
                  thinking_level: Optional[str], source: str, thinking_name: Optional[str]) -> str:
        """
        The one path for synchronous Gemini calls: streams the response, writes the thought
        summaries (if any, and if thinking_name is given), logs usage and cost from the final
        chunk, and returns the answer text only (block / finish-reason checks as for batch results).
        """
        thinking_kwargs = {"include_thoughts": True}
        if thinking_level:
            thinking_kwargs["thinking_level"] = thinking_level
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(**thinking_kwargs),
        )
        stream = self.client.models.generate_content_stream(
            model=self.model_name, contents=user_prompt, config=config
        )
        response, thoughts = _collect_stream(stream)
        if thoughts and thinking_name:
            save_output_file(directory=self.thinking_dir, base_name=f"{thinking_name}_thinking",
                             extension="txt", content=thoughts, logger=logger,
                             label=f"Thinking summary of {source}")
        log_gemini_usage(logger, response, self.model_name, self.pricing)
        return _gemini_response_text(response, source)

    def generate_siman_script(
        self, 
        siman: int, 
        master_context: str, 
        system_instruction: str,
        thinking_level: Optional[str] = None,
        thinking_name: Optional[str] = None,
    ) -> str:
        gematria_siman = int_to_gematria(siman)
        logger.info(f"Generating script for Siman {siman} ({gematria_siman}) using Gemini model {self.model_name}"
                    + (f" (thinking_level {thinking_level})" if thinking_level else "") + "...")
        user_prompt = self._get_generation_user_prompt(gematria_siman, master_context)
        try:
            script_text = self._generate(system_instruction, user_prompt, self.temperature, thinking_level,
                                         f"Gemini script generation for Siman {siman}", thinking_name)
        except Exception as e:
            logger.error(f"Error during Gemini script generation for Siman {siman}: {e}")
            raise
        logger.info(f"Successfully generated script for Siman {siman} ({len(script_text)} chars).")
        return script_text

    def analyze_cross_relations(self, drafts: dict, relations_instruction: str,
                                thinking_name: Optional[str] = None) -> str:
        logger.info(f"Analyzing cross-relations for {list(drafts.keys())} via Gemini model {self.model_name}...")
        user_prompt = self._get_relations_user_prompt(drafts)
        try:
            relations_text = self._generate(relations_instruction, user_prompt, self.temperature, None,
                                            "Gemini relations analysis", thinking_name)
        except Exception as e:
            logger.error(f"Error during Gemini relations analysis: {e}")
            raise
        logger.info("Successfully generated cross-relations analysis.")
        return relations_text

    def polish_siman_script(self, script_text: str, relations_text: str, polishing_instruction: str,
                            temperature: float,
                            thinking_name: Optional[str] = None) -> str:
        logger.info("Polishing generated script for TTS compliance via Gemini...")
        user_prompt = self._get_polishing_user_prompt(script_text, relations_text)
        try:
            polished_text = self._generate(polishing_instruction, user_prompt, temperature, None,
                                           "Gemini polish call", thinking_name)
        except Exception as e:
            logger.error(f"Error during Gemini script polish call: {e}")
            raise
        logger.info(f"Successfully polished script ({len(polished_text)} chars).")
        return polished_text

    def generate_text(self, system_instruction: str, user_prompt: str, thinking_level: str = None,
                      thinking_name: Optional[str] = None) -> str:
        logger.info(
            f"Running Gemini text call via model {self.model_name} (temperature {self.temperature}"
            + (f", thinking_level {thinking_level}" if thinking_level else "") + ")..."
        )
        return self._generate(system_instruction, user_prompt, self.temperature, thinking_level,
                              "Gemini text call", thinking_name)

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
        state = self.client.batches.get(name=job_id).state
        state = getattr(state, "name", str(state))
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
            return "completed"
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            return "failed"
        return "pending"

    def get_batch_result(self, job_id: str) -> str:
        """
        Text of the job's single inlined response. A job can SUCCEED while its request failed:
        the InlinedResponse then carries `error` (code / message) instead of `response`.
        Raises BatchResultError with the API's reason for any failed, blocked or empty result.
        """
        job = self.client.batches.get(name=job_id)
        responses = (job.dest.inlined_responses if job.dest else None) or []
        if not responses:
            raise BatchResultError(f"Gemini batch job {job_id} has no inlined response.")
        inlined = responses[0]
        if inlined.error is not None:
            raise BatchResultError(f"Gemini batch request failed (job {job_id}): {_format_job_error(inlined.error)}")
        if inlined.response is None:
            raise BatchResultError(f"Gemini batch job {job_id} returned neither a response nor an error.")
        try:
            return _gemini_response_text(inlined.response, f"Gemini batch job {job_id}")
        except ValueError as e:
            raise BatchResultError(str(e)) from e



def build_gemini_client(api_key: Optional[str], ssl_verify: bool) -> genai.Client:
    """The Gemini client for every Gemini text call (generator, enhancer)."""
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if not ssl_verify:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        kwargs["http_options"] = {"client_args": {"verify": False}}
    try:
        return genai.Client(**kwargs)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini Client: {e}")
        raise


def _collect_stream(stream) -> tuple:
    """
    Folds a generate_content_stream into one GenerateContentResponse (answer parts only) plus the
    joined thought-summary text. prompt_feedback, finish_reason / finish_message and usage_metadata
    are taken from the chunks that carry them (usage and the finish reason arrive on the last one).
    """
    answer_parts, thoughts = [], []
    prompt_feedback = finish_reason = finish_message = usage = None
    for chunk in stream:
        if chunk.prompt_feedback:
            prompt_feedback = chunk.prompt_feedback
        if chunk.usage_metadata:
            usage = chunk.usage_metadata
        if not chunk.candidates:
            continue
        candidate = chunk.candidates[0]
        if candidate.finish_reason:
            finish_reason = candidate.finish_reason
        if candidate.finish_message:
            finish_message = candidate.finish_message
        for part in (candidate.content.parts if candidate.content else None) or []:
            if not part.text:
                continue
            if part.thought:
                thoughts.append(part.text)
            else:
                answer_parts.append(types.Part(text=part.text))
    candidates = []
    if answer_parts or finish_reason or finish_message:
        candidates = [types.Candidate(
            content=types.Content(role="model", parts=answer_parts),
            finish_reason=finish_reason, finish_message=finish_message,
        )]
    response = types.GenerateContentResponse(
        candidates=candidates, prompt_feedback=prompt_feedback, usage_metadata=usage
    )
    return response, "".join(thoughts)

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
    """The answer text (thought parts excluded); ValueError naming the block / finish reason otherwise
    (sync calls and batch results; get_batch_result re-raises it as BatchResultError)."""
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
    def __init__(self, api_key: Optional[str], model_name: str, temperature: float, service_tier: Optional[str],
                 ssl_verify: bool):
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
        thinking_level: Optional[str] = None,
        thinking_name: Optional[str] = None,
    ) -> str:
        gematria_siman = int_to_gematria(siman)
        logger.info(f"Generating script for Siman {siman} ({gematria_siman}) using OpenAI model {self.model_name}...")
        user_prompt = self._get_generation_user_prompt(gematria_siman, master_context)

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

    def analyze_cross_relations(self, drafts: dict, relations_instruction: str,
                                thinking_name: Optional[str] = None) -> str:
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

    def polish_siman_script(self, script_text: str, relations_text: str, polishing_instruction: str,
                            temperature: float,
                            thinking_name: Optional[str] = None) -> str:
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
                kwargs["temperature"] = temperature
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

    def generate_text(self, system_instruction: str, user_prompt: str, thinking_level: str = None,
                      thinking_name: Optional[str] = None) -> str:
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
        then in the error file (or carries `error` / a non-200 response). Raises BatchResultError with
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
            raise BatchResultError(f"OpenAI batch {job_id} completed without an output or error file.")
        record = records[0]
        source = f"OpenAI batch {job_id}"
        error = record.get("error")
        if error:
            raise BatchResultError(f"{source}: request failed (error code {error.get('code')}: {error.get('message')})")
        response = record.get("response") or {}
        body = response.get("body") or {}
        if response.get("status_code") != 200:
            message = (body.get("error") or {}).get("message", "no message")
            raise BatchResultError(f"{source}: request failed (HTTP {response.get('status_code')}: {message})")
        choices = body.get("choices") or []
        if not choices:
            raise BatchResultError(f"{source}: response has no choices.")
        finish = choices[0].get("finish_reason")
        text = (choices[0].get("message") or {}).get("content")
        if finish not in (None, "stop"):
            raise BatchResultError(f"{source}: generation stopped early (finish_reason {finish})")
        if not text:
            raise BatchResultError(f"{source}: response has no text (finish_reason {finish}).")
        return text
