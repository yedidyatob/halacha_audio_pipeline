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
        self.gemini_system_instruction = self.config_data.get("system_instruction") or self.config_data.get("gemini", {}).get("system_instruction", "")
        self.polishing_instruction = self.config_data.get("polishing_instruction") or ""
        self.relations_instruction = self.config_data.get("relations_instruction") or ""

        # TTS settings
        tts = self.config_data.get("tts", {})
        self.tts_engine = tts.get("engine", "elevenlabs").lower()
        if self.tts_engine not in ["elevenlabs", "google", "openai"]:
            raise ValueError(f"Invalid TTS engine choice: '{self.tts_engine}'. Must be 'elevenlabs', 'google', or 'openai'.")

        self.elevenlabs_settings = tts.get("elevenlabs", {})
        self.google_tts_settings = tts.get("google", {})
        self.openai_tts_settings = tts.get("openai", {})

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
