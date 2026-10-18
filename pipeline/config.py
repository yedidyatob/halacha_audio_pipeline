import os
import yaml
from typing import Dict, Any
from pipeline.logger import get_logger
from pipeline.domain import SECTIONS_METADATA
from pipeline.tts import (
    validate_gap_ms, validate_chunk_size, validate_context_chars,
    GOOGLE_TTS_MAX_INPUT_BYTES, GOOGLE_TTS_ASSUMED_BYTES_PER_CHAR,
)

logger = get_logger(__name__)

def _max_retries(value, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"'{key}' must be a non-negative integer, got {value!r}.")
    return value


class PipelineConfig:
    """
    Loads, parses, and validates the configuration file (config.yaml).
    Provides default values and validates credentials.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_data = self._load_yaml()
        self._validate_and_setup()

    def _load_yaml(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
                return data if data else {}
            except yaml.YAMLError as e:
                logger.error(f"Error parsing configuration YAML: {e}")
                raise

    def _validate_and_setup(self) -> None:
        # Section validation
        self.halachic_section = self.config_data.get("halachic_section")
        if not self.halachic_section:
            raise ValueError("Configuration must specify 'halachic_section'.")

        if self.halachic_section not in SECTIONS_METADATA:
            raise ValueError(
                f"Unsupported halachic_section: '{self.halachic_section}'. "
                f"Supported sections are: {list(SECTIONS_METADATA.keys())}"
            )
        self.section_metadata = SECTIONS_METADATA[self.halachic_section]
        self.section_slug = self.section_metadata["slug"]

        # Load API keys (YAML takes priority, fallback to environment variables)
        api_keys = self.config_data.get("api_keys", {})
        
        self.gemini_api_key = api_keys.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
        self.openai_api_key = api_keys.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        self.elevenlabs_api_key = api_keys.get("elevenlabs_api_key") or os.environ.get("ELEVENLABS_API_KEY")
        self.google_tts_credentials = api_keys.get("google_tts_credentials") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Sefaria settings
        sefaria = self.config_data.get("sefaria", {})
        self.sefaria_base_url = sefaria.get("base_url", "https://www.sefaria.org/api")
        self.sefaria_timeout = sefaria.get("timeout_seconds", 15)
        self.sefaria_retries = sefaria.get("retries", 3)
        self.ssl_verify = sefaria.get("ssl_verify", True)

        # Generator settings (unified or fallback)
        generator = self.config_data.get("generator", {})
        self.generator_engine = generator.get("engine", "gemini").lower()
        if self.generator_engine not in ["gemini", "openai"]:
            raise ValueError(f"Invalid generator engine choice: '{self.generator_engine}'. Must be 'gemini' or 'openai'.")

        # Load gemini generator settings
        gemini_settings = generator.get("gemini", {})
        self.gemini_model_name = gemini_settings.get("model_name", "gemini-3.5-flash")
        self.gemini_temperature = gemini_settings.get("temperature", 0.3)

        # Load openai generator settings
        openai_settings = generator.get("openai", {})
        self.openai_model_name = openai_settings.get("model_name", "o1")
        self.openai_temperature = openai_settings.get("temperature", 1.0)
        self.openai_service_tier = openai_settings.get("service_tier", "flex")

        # Load system instruction (shared at root or falling back to gemini section)
        raw_system_instruction = self.config_data.get("system_instruction") or self.config_data.get("gemini", {}).get("system_instruction", "")
        self.gemini_system_instruction = raw_system_instruction.format(
            commentators_list_hebrew=self.section_metadata.get("commentators_list_hebrew", ""),
            prompt_commentators_desc=self.section_metadata.get("prompt_commentators_desc", "")
        ) if raw_system_instruction else ""
        
        raw_polishing_instruction = self.config_data.get("polishing_instruction") or ""
        self.polishing_instruction = raw_polishing_instruction.format(
            commentators_list_short=self.section_metadata.get("commentators_list_short", ""),
            tts_abbreviations=self.section_metadata.get("tts_abbreviations", "")
        ) if raw_polishing_instruction else ""
        
        raw_relations_instruction = self.config_data.get("relations_instruction") or ""
        self.relations_instruction = raw_relations_instruction.format(
            hebrew_name=self.section_metadata.get("hebrew_name", ""),
            prompt_commentators_desc=self.section_metadata.get("prompt_commentators_desc", "")
        ) if raw_relations_instruction else ""


        # TTS settings
        tts = self.config_data.get("tts", {})
        self.tts_engine = tts.get("engine", "elevenlabs").lower()
        if self.tts_engine not in ["elevenlabs", "google", "openai", "gemini"]:
            raise ValueError(f"Invalid TTS engine choice: '{self.tts_engine}'. Must be 'elevenlabs', 'google', 'openai', or 'gemini'.")

        self.elevenlabs_settings = tts.get("elevenlabs", {})
        self.google_tts_settings = tts.get("google", {})
        self.openai_tts_settings = tts.get("openai", {})
        self.gemini_tts_settings = tts.get("gemini", {})

        # Silence inserted between synthesized chunks (ms). Required: config.yaml is the
        # single source of truth. Set both to 0 to disable (chunks are stream-copied).
        for key, attr in (
            ("chunk_gap_paragraph_ms", "tts_chunk_gap_paragraph_ms"),
            ("chunk_gap_sentence_ms", "tts_chunk_gap_sentence_ms"),
        ):
            if tts.get(key) is None:
                raise ValueError(f"Missing required configuration parameter '{key}' under 'tts' in config.yaml.")
            setattr(self, attr, validate_gap_ms(tts[key], f"tts.{key}"))
        
        # Max characters per TTS request chunk for the selected engine (tts.<engine>.chunk_size).
        # Required: no defaults in code.
        engine_settings = tts.get(self.tts_engine) or {}
        if engine_settings.get("chunk_size") is None:
            raise ValueError(
                f"Missing required configuration parameter 'chunk_size' under 'tts.{self.tts_engine}' in config.yaml."
            )
        self.tts_chunk_size = validate_chunk_size(engine_settings["chunk_size"], f"tts.{self.tts_engine}.chunk_size")
        if self.tts_engine == "google":
            max_chars = GOOGLE_TTS_MAX_INPUT_BYTES // GOOGLE_TTS_ASSUMED_BYTES_PER_CHAR
            if self.tts_chunk_size > max_chars:
                raise ValueError(
                    f"'tts.google.chunk_size' is {self.tts_chunk_size}, but Google TTS accepts at most "
                    f"{GOOGLE_TTS_MAX_INPUT_BYTES} bytes per request ({max_chars} Hebrew characters)."
                )
        if self.tts_engine == "elevenlabs":
            if self.elevenlabs_settings.get("context_chars") is None:
                raise ValueError(
                    "Missing required configuration parameter 'context_chars' under 'tts.elevenlabs' in config.yaml."
                )
            self.elevenlabs_context_chars = validate_context_chars(
                self.elevenlabs_settings["context_chars"], "tts.elevenlabs.context_chars"
            )

        # Validation for Gemini TTS settings (no hardcoded defaults allowed in pipeline)
        if self.tts_engine == "gemini":
            if not self.gemini_tts_settings:
                raise ValueError("TTS engine is set to 'gemini' but the 'gemini' configuration section is missing under 'tts' in config.yaml.")
            self.gemini_tts_model = self.gemini_tts_settings.get("model_name")
            if not self.gemini_tts_model:
                raise ValueError("Missing required configuration parameter 'model_name' under 'tts.gemini' in config.yaml.")
            
            raw_voice = self.gemini_tts_settings.get("voice_name")
            if not raw_voice:
                raise ValueError("Missing required configuration parameter 'voice_name' under 'tts.gemini' in config.yaml.")
            
            # Map/clean legacy Google voice names to Gemini prebuilt voices
            if "Achird" in raw_voice:
                self.gemini_tts_voice = "Achird"
            elif "Achernar" in raw_voice:
                self.gemini_tts_voice = "Aoede"
            elif raw_voice in ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Achird"]:
                self.gemini_tts_voice = raw_voice
            else:
                raise ValueError(
                    f"Unsupported Gemini TTS voice: '{raw_voice}'. "
                    f"Supported prebuilt voices are: Puck, Charon, Kore, Fenrir, Aoede, Achird."
                )
                
        # Directory setup
        dirs = self.config_data.get("directories", {})
        self.output_dir = dirs.get("output_dir", "./output")
        self.cache_dir = dirs.get("cache_dir", "./cache")
        self.drafts_dir = os.path.join(self.output_dir, "drafts")
        self.relations_dir = os.path.join(self.output_dir, "relations")

        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.drafts_dir, exist_ok=True)
        os.makedirs(self.relations_dir, exist_ok=True)
        
        logger.info("Configuration validated and directories established.")

    @property
    def generator_max_retries(self) -> int:
        """
        Extra synchronous calls after a failed generator attempt in Stages 1-3 (API error, failed
        batch job or request, empty / blocked / cut-off response). Required: no code default.
        """
        value = (self.config_data.get("generator") or {}).get("max_retries")
        if value is None:
            raise ValueError("Missing required configuration parameter 'max_retries' under 'generator' in config.yaml.")
        return _max_retries(value, "generator.max_retries")

    # ------------------------------------------------------------------
    # Pronunciation stage (LLM nikkud for ambiguous words). No defaults: every value
    # must come from config.yaml; a missing key raises a clear ValueError.
    # ------------------------------------------------------------------
    def _pronunciation_section(self) -> Dict[str, Any]:
        section = self.config_data.get("pronunciation")
        if not isinstance(section, dict):
            raise ValueError("Missing required configuration section 'pronunciation' in config.yaml.")
        return section

    def _pronunciation_value(self, key: str):
        section = self._pronunciation_section()
        if section.get(key) is None:
            raise ValueError(f"Missing required configuration parameter '{key}' under 'pronunciation' in config.yaml.")
        return section[key]

    @property
    def pronunciation_enabled(self) -> bool:
        value = self._pronunciation_value("enabled")
        if not isinstance(value, bool):
            raise ValueError(f"'pronunciation.enabled' must be true or false, got {value!r}.")
        return value

    @property
    def pronunciation_engine(self) -> str:
        engine = str(self._pronunciation_value("engine")).lower()
        if engine not in ("gemini", "openai"):
            raise ValueError(f"Invalid pronunciation engine: '{engine}'. Must be 'gemini' or 'openai'.")
        return engine

    @property
    def pronunciation_model_name(self) -> str:
        return self._pronunciation_value("model_name")

    @property
    def pronunciation_temperature(self) -> float:
        value = self._pronunciation_value("temperature")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'pronunciation.temperature' must be a number, got {value!r}.")
        return float(value)

    @property
    def pronunciation_thinking_level(self) -> str:
        """Gemini thinking level for the pronunciation call. Required for engine gemini; ignored for openai."""
        if self.pronunciation_engine != "gemini":
            return None
        value = self._pronunciation_value("thinking_level")
        if value not in ("low", "medium", "high"):
            raise ValueError(f"'pronunciation.thinking_level' must be one of low, medium, high, got {value!r}.")
        return value

    @property
    def pronunciation_max_retries(self) -> int:
        """Extra LLM calls after a failed pronunciation attempt (LLM error or rejected output)."""
        return _max_retries(self._pronunciation_value("max_retries"), "pronunciation.max_retries")

    def _pronunciation_word_list(self, key: str) -> list:
        value = self._pronunciation_value(key)
        if not isinstance(value, list) or not all(isinstance(w, str) and w.strip() for w in value):
            raise ValueError(f"'pronunciation.{key}' must be a list of non-empty strings, got {value!r}.")
        return [w.strip() for w in value]

    @property
    def pronunciation_always_vocalize(self) -> list:
        return self._pronunciation_word_list("always_vocalize")

    @property
    def pronunciation_never_vocalize(self) -> list:
        return self._pronunciation_word_list("never_vocalize")

    @staticmethod
    def _render_word_list(words: list) -> str:
        return ", ".join(words) if words else "אין"

    @property
    def pronunciation_instruction(self) -> str:
        """The prompt with {always_vocalize} / {never_vocalize} filled in (targeted replace; other braces untouched)."""
        value = self.config_data.get("pronunciation_instruction")
        if not value or not str(value).strip():
            raise ValueError("Missing required configuration parameter 'pronunciation_instruction' in config.yaml.")
        return (str(value)
                .replace("{always_vocalize}", self._render_word_list(self.pronunciation_always_vocalize))
                .replace("{never_vocalize}", self._render_word_list(self.pronunciation_never_vocalize)))
