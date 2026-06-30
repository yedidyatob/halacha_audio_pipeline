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
    """
    if config.tts_engine == "elevenlabs":
        return ElevenLabsTTS(
            api_key=config.elevenlabs_api_key,
            voice_id=config.elevenlabs_settings.get("voice_id", "Adam"),
            model_id=config.elevenlabs_settings.get("model_id", "eleven_multilingual_v3"),
            stability=config.elevenlabs_settings.get("stability", 0.5),
            similarity_boost=config.elevenlabs_settings.get("similarity_boost", 0.75),
            ssl_verify=config.ssl_verify
        )
    elif config.tts_engine == "google":
        return GoogleCloudTTS(
            credentials_path=config.google_tts_credentials,
            voice_name=config.google_tts_settings.get("voice_name", "he-IL-Neural2-M"),
            language_code=config.google_tts_settings.get("language_code", "he-IL"),
            speaking_rate=config.google_tts_settings.get("speaking_rate", 1.0),
            pitch=config.google_tts_settings.get("pitch", 0.0)
        )
    elif config.tts_engine == "openai":
        return OpenAITTS(
            api_key=config.openai_api_key,
            voice=config.openai_tts_settings.get("voice", "alloy"),
            model=config.openai_tts_settings.get("model", "tts-1"),
            speed=config.openai_tts_settings.get("speed", 1.0),
            ssl_verify=config.ssl_verify
        )
    elif config.tts_engine == "gemini":
        return GeminiTTS(
            model_id=config.gemini_tts_model,
            voice_name=config.gemini_tts_voice,
            temperature=config.gemini_tts_temp,
            api_key=config.gemini_api_key
        )
    else:
        raise ValueError(f"Unsupported TTS engine: {config.tts_engine}")
