import os
import yaml
from typing import Dict, Any
from pipeline.logger import get_logger
from pipeline.domain import SECTIONS_METADATA
from pipeline.nikkud_review import has_nikkud, word_vocalization_error
from pipeline.tts import (
    validate_gap_ms, validate_chunk_size, validate_context_chars,
    GOOGLE_TTS_MAX_INPUT_BYTES, GOOGLE_TTS_ASSUMED_BYTES_PER_CHAR,
)

logger = get_logger(__name__)

def _max_retries(value, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"'{key}' must be a non-negative integer, got {value!r}.")
    return value


def _temperature(value, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a number, got {value!r}.")
    return float(value)


def _required(section: dict, key: str, path: str):
    """section[key], or a ValueError naming the missing 'key' under 'path' in config.yaml."""
    value = (section or {}).get(key)
    if value is None:
        raise ValueError(f"Missing required configuration parameter '{key}' under '{path}' in config.yaml.")
    return value


def _required_str(section: dict, key: str, path: str) -> str:
    value = _required(section, key, path)
    if not _non_empty_str(value):
        raise ValueError(f"'{path}.{key}' must be a non-empty string, got {value!r}.")
    return value


THINKING_LEVELS = ("low", "medium", "high")


def _number_in_range(value, key: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"'{key}' must be a number between {low} and {high}, got {value!r}.")
    return value


def _positive_int(value, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"'{key}' must be a positive integer, got {value!r}.")
    return value


def _bool(value, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be true or false, got {value!r}.")
    return value


def _engine(section: dict, path: str, choices: tuple) -> str:
    value = _required_str(section, "engine", path).lower()
    if value not in choices:
        raise ValueError(f"Invalid {path}.engine choice: '{value}'. Must be one of: {', '.join(choices)}.")
    return value


def _non_empty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _always_vocalize_entry(entry, key: str) -> str:
    """
    One pronunciation.always_vocalize entry, rendered for the prompt: a plain word as is, or
    {word, readings: [{nikkud, meaning}, ...]} as 'word (nikkud = meaning; ...)'. Each reading's
    nikkud must be a valid vocalization of the word (same letters, ו / י may be added or removed
    except as the first letter), so a typo is caught at startup.
    """
    if isinstance(entry, str):
        if not entry.strip():
            raise ValueError(f"'{key}' entries must be non-empty, got {entry!r}.")
        return entry.strip()
    if not isinstance(entry, dict) or set(entry) != {"word", "readings"}:
        raise ValueError(f"'{key}' entries must be a word or a mapping with 'word' and 'readings', got {entry!r}.")
    word, readings = entry["word"], entry["readings"]
    if not _non_empty_str(word) or has_nikkud(word):
        raise ValueError(f"'{key}' entry 'word' must be a non-empty word without nikkud, got {word!r}.")
    word = word.strip()
    if not isinstance(readings, list) or not readings:
        raise ValueError(f"'{key}' entry {word!r}: 'readings' must be a non-empty list, got {readings!r}.")
    rendered = []
    for reading in readings:
        if not isinstance(reading, dict) or set(reading) != {"nikkud", "meaning"}:
            raise ValueError(f"'{key}' entry {word!r}: each reading must be a mapping with 'nikkud' and "
                             f"'meaning', got {reading!r}.")
        nikkud, meaning = reading["nikkud"], reading["meaning"]
        if not _non_empty_str(meaning):
            raise ValueError(f"'{key}' entry {word!r}: reading {nikkud!r} needs a non-empty 'meaning'.")
        if not _non_empty_str(nikkud) or not has_nikkud(nikkud):
            raise ValueError(f"'{key}' entry {word!r}: reading 'nikkud' must be the vocalized word, got {nikkud!r}.")
        error = word_vocalization_error(word, nikkud.strip())
        if error:
            raise ValueError(f"'{key}' entry {word!r}: reading {nikkud!r} is not a vocalization of the word ({error}).")
        rendered.append(f"{nikkud.strip()} = {meaning.strip()}")
    return f"{word} ({'; '.join(rendered)})"


def _rates(entry: dict, key: str) -> dict:
    rates = {}
    for name in ("input_per_million", "output_per_million"):
        value = entry.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"'{key}.{name}' must be a non-negative number (USD per 1M tokens), got {value!r}.")
        rates[name] = float(value)
    return rates


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
        
        # Sefaria settings (required; no code defaults)
        sefaria = self.config_data.get("sefaria")
        self.sefaria_base_url = _required_str(sefaria, "base_url", "sefaria")
        timeout = _required(sefaria, "timeout_seconds", "sefaria")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"'sefaria.timeout_seconds' must be a positive number, got {timeout!r}.")
        self.sefaria_timeout = timeout
        # attempts per request (the extractor tries each request this many times)
        self.sefaria_retries = _positive_int(_required(sefaria, "retries", "sefaria"), "sefaria.retries")
        self.ssl_verify = _bool(_required(sefaria, "ssl_verify", "sefaria"), "sefaria.ssl_verify")

        # Generator settings (unified or fallback)
        self.generator_engine = _engine(self.config_data.get("generator"), "generator", ("gemini", "openai"))

        # Load system instruction (shared at root or falling back to gemini section)
        raw_system_instruction = self.config_data.get("system_instruction") or self.config_data.get("gemini", {}).get("system_instruction", "")
        self.gemini_system_instruction = raw_system_instruction.format(
            commentators_list_hebrew=self.section_metadata.get("commentators_list_hebrew", ""),
            prompt_commentators_desc=self.section_metadata.get("prompt_commentators_desc", "")
        ) if raw_system_instruction else ""
        
        
        raw_relations_instruction = self.config_data.get("relations_instruction") or ""
        self.relations_instruction = raw_relations_instruction.format(
            hebrew_name=self.section_metadata.get("hebrew_name", ""),
            prompt_commentators_desc=self.section_metadata.get("prompt_commentators_desc", "")
        ) if raw_relations_instruction else ""


        # TTS settings
        tts = self.config_data.get("tts") or {}
        self.tts_engine = _engine(tts, "tts", ("elevenlabs", "google", "openai", "gemini"))

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
            # Cloud Text-to-Speech request fields (audioConfig.pitch / speakingRate, voice.languageCode)
            self.gemini_tts_pitch = _number_in_range(
                _required(self.gemini_tts_settings, "pitch", "tts.gemini"), "tts.gemini.pitch", -20.0, 20.0)
            self.gemini_tts_speaking_rate = _number_in_range(
                _required(self.gemini_tts_settings, "speaking_rate", "tts.gemini"), "tts.gemini.speaking_rate", 0.25, 2.0)
            self.gemini_tts_language_code = _required_str(self.gemini_tts_settings, "language_code", "tts.gemini")
                
        # Directory setup
        dirs = self.config_data.get("directories", {})
        self.output_dir = dirs.get("output_dir", "./output")
        self.cache_dir = dirs.get("cache_dir", "./cache")
        self.drafts_dir = os.path.join(self.output_dir, "drafts")
        if not dirs.get("thinking_dir"):
            raise ValueError("Missing required configuration parameter 'thinking_dir' under 'directories' in config.yaml.")
        self.thinking_dir = dirs["thinking_dir"]
        self.relations_dir = os.path.join(self.output_dir, "relations")

        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.drafts_dir, exist_ok=True)
        os.makedirs(self.relations_dir, exist_ok=True)
        os.makedirs(self.thinking_dir, exist_ok=True)
        
        logger.info("Configuration validated and directories established.")

    @property
    def generator_max_retries(self) -> int:
        """
        Retry rounds after a failed generator attempt in Stages 1-3 (API error, failed batch job or
        request, empty / blocked / cut-off response): new sync calls, or new batch jobs in batch
        mode. Required: no code default.
        """
        value = (self.config_data.get("generator") or {}).get("max_retries")
        if value is None:
            raise ValueError("Missing required configuration parameter 'max_retries' under 'generator' in config.yaml.")
        return _max_retries(value, "generator.max_retries")

    @property
    def polish_temperature(self) -> float:
        """Temperature for the Stage 3 polish calls, sync and batch. Required: no code default."""
        value = _required(self.config_data.get("generator"), "polish_temperature", "generator")
        return _temperature(value, "generator.polish_temperature")

    def _generator_engine_section(self, engine: str) -> dict:
        return (self.config_data.get("generator") or {}).get(engine)

    @property
    def gemini_model_name(self) -> str:
        return _required_str(self._generator_engine_section("gemini"), "model_name", "generator.gemini")

    @property
    def gemini_temperature(self) -> float:
        value = _required(self._generator_engine_section("gemini"), "temperature", "generator.gemini")
        return _temperature(value, "generator.gemini.temperature")

    @property
    def openai_model_name(self) -> str:
        return _required_str(self._generator_engine_section("openai"), "model_name", "generator.openai")

    @property
    def openai_temperature(self) -> float:
        """Ignored by OpenAI reasoning models (o1/o3), but still required."""
        value = _required(self._generator_engine_section("openai"), "temperature", "generator.openai")
        return _temperature(value, "generator.openai.temperature")

    @property
    def openai_service_tier(self) -> str:
        """The generator's OpenAI service tier (engine openai)."""
        return _required_str(self._generator_engine_section("openai"), "service_tier", "generator.openai")

    def _native_audio_section(self) -> dict:
        return (self.config_data.get("experiments") or {}).get("gemini_native_audio")

    @property
    def native_audio_model_name(self) -> str:
        """Model for the scripts/ Gemini native-audio experiments."""
        return _required_str(self._native_audio_section(), "model_name", "experiments.gemini_native_audio")

    @property
    def native_audio_voice_name(self) -> str:
        return _required_str(self._native_audio_section(), "voice_name", "experiments.gemini_native_audio")

    @property
    def native_audio_temperature(self) -> float:
        value = _required(self._native_audio_section(), "temperature", "experiments.gemini_native_audio")
        return _temperature(value, "experiments.gemini_native_audio.temperature")

    @property
    def draft_thinking_level(self) -> str:
        """Gemini thinking level for the Stage 1 draft calls (sync and batch). Required for engine gemini."""
        if self.generator_engine != "gemini":
            return None
        value = (self.config_data.get("generator") or {}).get("draft_thinking_level")
        if value is None:
            raise ValueError("Missing required configuration parameter 'draft_thinking_level' under 'generator' in config.yaml.")
        if value not in THINKING_LEVELS:
            raise ValueError(f"'generator.draft_thinking_level' must be one of low, medium, high, got {value!r}.")
        return value

    def gemini_price(self, model_name: str) -> dict:
        """
        The validated gemini_pricing entry for a model (USD per 1M tokens): input_per_million,
        output_per_million (thinking tokens are billed as output) and an optional long_context tier
        {threshold_tokens, input_per_million, output_per_million}. A model without an entry is an error.
        """
        section = self.config_data.get("gemini_pricing")
        if not isinstance(section, dict):
            raise ValueError("Missing required configuration section 'gemini_pricing' in config.yaml.")
        entry = section.get(model_name)
        if not isinstance(entry, dict):
            raise ValueError(f"No price for Gemini model '{model_name}' under 'gemini_pricing' in config.yaml.")
        key = f"gemini_pricing.{model_name}"
        price = _rates(entry, key)
        tier = entry.get("long_context")
        if tier is not None:
            if not isinstance(tier, dict):
                raise ValueError(f"'{key}.long_context' must be a mapping, got {tier!r}.")
            threshold = tier.get("threshold_tokens")
            if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
                raise ValueError(f"'{key}.long_context.threshold_tokens' must be a positive integer, got {threshold!r}.")
            price["long_context"] = {"threshold_tokens": threshold, **_rates(tier, f"{key}.long_context")}
        unknown = set(entry) - {"input_per_million", "output_per_million", "long_context"}
        if unknown:
            raise ValueError(f"Unknown key(s) under '{key}' in config.yaml: {sorted(unknown)}.")
        return price

    # ------------------------------------------------------------------
    # Stage 3 polish + nikkud prompt. No defaults: every value must come from config.yaml;
    # a missing key raises a clear ValueError.
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
    def pronunciation_always_vocalize(self) -> list:
        """The always_vocalize entries as rendered in the prompt (see _always_vocalize_entry)."""
        value = self._pronunciation_value("always_vocalize")
        if not isinstance(value, list):
            raise ValueError(f"'pronunciation.always_vocalize' must be a list, got {value!r}.")
        return [_always_vocalize_entry(entry, "pronunciation.always_vocalize") for entry in value]

    @property
    def pronunciation_never_vocalize(self) -> list:
        value = self._pronunciation_value("never_vocalize")
        if not isinstance(value, list) or not all(_non_empty_str(w) for w in value):
            raise ValueError(f"'pronunciation.never_vocalize' must be a list of non-empty strings, got {value!r}.")
        return [w.strip() for w in value]

    @staticmethod
    def _render_word_list(words: list) -> str:
        return ", ".join(words) if words else "אין"

    @property
    def single_pass_user_prompt(self) -> str:
        """
        --single-pass: the Stage 3 user prompt for a siman's raw Sefaria sources (instead of a Stage 1 draft),
        with {gematria_siman} and {sources} placeholders (fill with format_single_pass_user_prompt).
        """
        value = self.config_data.get("single_pass_user_prompt")
        if not _non_empty_str(value):
            raise ValueError("Missing required configuration parameter 'single_pass_user_prompt' in config.yaml.")
        for placeholder in ("{gematria_siman}", "{sources}"):
            if placeholder not in value:
                raise ValueError(f"'single_pass_user_prompt' in config.yaml must contain {placeholder}.")
        return value

    def format_single_pass_user_prompt(self, gematria_siman: str, sources: str) -> str:
        return self.single_pass_user_prompt.format(gematria_siman=gematria_siman, sources=sources)

    @property
    def polishing_instruction(self) -> str:
        """
        The Stage 3 system instruction (polish + nikkud): polishing_instruction with the section's
        {commentators_list_short} / {tts_abbreviations} and the pronunciation lists
        {always_vocalize} / {never_vocalize} filled in.
        """
        value = self.config_data.get("polishing_instruction")
        if not _non_empty_str(value):
            raise ValueError("Missing required configuration parameter 'polishing_instruction' in config.yaml.")
        return value.format(
            commentators_list_short=self.section_metadata.get("commentators_list_short", ""),
            tts_abbreviations=self.section_metadata.get("tts_abbreviations", ""),
            always_vocalize=self._render_word_list(self.pronunciation_always_vocalize),
            never_vocalize=self._render_word_list(self.pronunciation_never_vocalize),
        )