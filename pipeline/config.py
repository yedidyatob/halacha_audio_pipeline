import os
import yaml
from typing import Dict, Any
from pipeline.logger import get_logger
from pipeline.domain import SECTIONS_METADATA

logger = get_logger(__name__)

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
                
            self.gemini_tts_temp = self.gemini_tts_settings.get("temperature")
            if self.gemini_tts_temp is None:
                raise ValueError("Missing required configuration parameter 'temperature' under 'tts.gemini' in config.yaml.")

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
