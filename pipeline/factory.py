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
            temperature=config.gemini_temperature,
            ssl_verify=config.ssl_verify,
            pricing=config.gemini_price(config.gemini_model_name),
            thinking_dir=config.thinking_dir,
        )
    elif config.generator_engine == "openai":
        return OpenAIScriptGenerator(
            api_key=config.openai_api_key,
            model_name=config.openai_model_name,
            temperature=config.openai_temperature,
            service_tier=config.openai_service_tier,
            ssl_verify=config.ssl_verify
        )
    else:
        raise ValueError(f"Unsupported Generator engine: {config.generator_engine}")

def create_tts_engine(config: PipelineConfig) -> BaseTTS:
    """
    Factory function to return the configured TTS Synthesizer engine.
    All engine-specific settings come from config.yaml (single source of truth).
    """
    # Silence between synthesized chunks and the saved-chunk cleanup (shared by all engines).
    shared = dict(
        chunk_gap_paragraph_ms=config.tts_chunk_gap_paragraph_ms,
        chunk_gap_sentence_ms=config.tts_chunk_gap_sentence_ms,
        keep_chunks=config.tts_keep_chunks,
    )
    if config.tts_engine == "elevenlabs":
        el = config.elevenlabs_settings
        return ElevenLabsTTS(
            api_key=config.elevenlabs_api_key,
            voice_id=el["voice_id"],
            model_id=el["model_id"],
            stability=el["stability"],
            similarity_boost=el["similarity_boost"],
            ssl_verify=config.ssl_verify,
            chunk_size=config.tts_chunk_size,
            context_chars=config.elevenlabs_context_chars,
            language_code=config.elevenlabs_language_code,
            request_timeout_seconds=config.tts_request_timeout_seconds,
            **shared
        )
    elif config.tts_engine == "google":
        g = config.google_tts_settings
        return GoogleCloudTTS(
            credentials_path=config.google_tts_credentials,
            voice_name=g["voice_name"],
            language_code=g["language_code"],
            speaking_rate=g["speaking_rate"],
            pitch=g["pitch"],
            chunk_size=config.tts_chunk_size,
            **shared
        )
    elif config.tts_engine == "openai":
        o = config.openai_tts_settings
        return OpenAITTS(
            api_key=config.openai_api_key,
            voice=o["voice"],
            model=o["model"],
            speed=o["speed"],
            ssl_verify=config.ssl_verify,
            chunk_size=config.tts_chunk_size,
            request_timeout_seconds=config.tts_request_timeout_seconds,
            **shared
        )
    elif config.tts_engine == "gemini":
        return GeminiTTS(
            api_key=config.gemini_api_key,
            model=config.gemini_tts_model,
            voice=config.gemini_tts_voice,
            language=config.gemini_tts_language,
            style=config.gemini_tts_style,
            ssl_verify=config.ssl_verify,
            chunk_size=config.tts_chunk_size,
            **shared
        )
    else:
        raise ValueError(f"Unsupported TTS engine: {config.tts_engine}")
