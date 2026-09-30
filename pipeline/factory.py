from pipeline.config import PipelineConfig
from pipeline.generator import BaseScriptGenerator, GeminiScriptGenerator, OpenAIScriptGenerator
from pipeline.tts import BaseTTS, ElevenLabsTTS, GoogleCloudTTS, OpenAITTS, GeminiTTS

def create_generator_engine(config: PipelineConfig) -> BaseScriptGenerator:
    """
    Factory function to return the configured Script Generator engine.
    """
    if config.generator_engine == "gemini":
        return GeminiScriptGenerator(
            api_key=config.gemini_api_key,
            model_name=config.gemini_model_name,
            temperature=config.gemini_temperature
        )
    elif config.generator_engine == "openai":
        return OpenAIScriptGenerator(
            api_key=config.openai_api_key,
            model_name=config.openai_model_name,
            temperature=config.openai_temperature,
            service_tier=config.openai_service_tier
        )
    else:
        raise ValueError(f"Unsupported Generator engine: {config.generator_engine}")

def create_tts_engine(config: PipelineConfig) -> BaseTTS:
    """
    Factory function to return the configured TTS Synthesizer engine.
    All engine-specific settings come from config.yaml (single source of truth).
    """
    if config.tts_engine == "elevenlabs":
        el = config.elevenlabs_settings
        return ElevenLabsTTS(
            api_key=config.elevenlabs_api_key,
            voice_id=el["voice_id"],
            model_id=el["model_id"],
            stability=el["stability"],
            similarity_boost=el["similarity_boost"],
            ssl_verify=config.ssl_verify
        )
    elif config.tts_engine == "google":
        g = config.google_tts_settings
        return GoogleCloudTTS(
            credentials_path=config.google_tts_credentials,
            voice_name=g["voice_name"],
            language_code=g["language_code"],
            speaking_rate=g.get("speaking_rate", 1.0),
            pitch=g.get("pitch", 0.0)
        )
    elif config.tts_engine == "openai":
        o = config.openai_tts_settings
        return OpenAITTS(
            api_key=config.openai_api_key,
            voice=o["voice"],
            model=o["model"],
            speed=o.get("speed", 1.0),
            ssl_verify=config.ssl_verify
        )
    elif config.tts_engine == "gemini":
        return GeminiTTS(
            model_id=config.gemini_tts_model,
            voice_name=config.gemini_tts_voice,
            api_key=config.gemini_api_key
        )
    else:
        raise ValueError(f"Unsupported TTS engine: {config.tts_engine}")
