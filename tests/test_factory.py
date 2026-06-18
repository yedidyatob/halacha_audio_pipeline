import pytest
from unittest.mock import patch, MagicMock
from pipeline.config import PipelineConfig
from pipeline.factory import create_generator_engine, create_tts_engine
from pipeline.generator import GeminiScriptGenerator, OpenAIScriptGenerator
from pipeline.tts import ElevenLabsTTS, GoogleCloudTTS, OpenAITTS

@patch("google.genai.Client")
def test_create_generator_engine_gemini(mock_client, tmp_path):
    config = MagicMock(spec=PipelineConfig)
    config.generator_engine = "gemini"
    config.gemini_api_key = "fake-key"
    config.gemini_model_name = "gemini-3.5-flash"
    config.gemini_temperature = 0.3
    
    engine = create_generator_engine(config)
    assert isinstance(engine, GeminiScriptGenerator)
    assert engine.model_name == "gemini-3.5-flash"

@patch("pipeline.generator.OpenAI")
def test_create_generator_engine_openai(mock_client, tmp_path):
    config = MagicMock(spec=PipelineConfig)
    config.generator_engine = "openai"
    config.openai_api_key = "fake-key"
    config.openai_model_name = "gpt-4o"
    config.openai_temperature = 0.7
    config.openai_service_tier = "flex"
    
    engine = create_generator_engine(config)
    assert isinstance(engine, OpenAIScriptGenerator)
    assert engine.model_name == "gpt-4o"

def test_create_generator_engine_invalid():
    config = MagicMock(spec=PipelineConfig)
    config.generator_engine = "invalid_engine"
    with pytest.raises(ValueError, match="Unsupported Generator engine"):
        create_generator_engine(config)

@patch("elevenlabs.client.ElevenLabs")
def test_create_tts_engine_elevenlabs(mock_elevenlabs):
    config = MagicMock(spec=PipelineConfig)
    config.tts_engine = "elevenlabs"
    config.elevenlabs_api_key = "fake-key"
    config.elevenlabs_settings = {
        "voice_id": "Adam",
        "model_id": "eleven_multilingual_v3",
        "stability": 0.5,
        "similarity_boost": 0.75
    }
    config.ssl_verify = True
    
    engine = create_tts_engine(config)
    assert isinstance(engine, ElevenLabsTTS)
    assert engine.voice_id == "Adam"

@patch("google.cloud.texttospeech.TextToSpeechClient")
def test_create_tts_engine_google(mock_google):
    config = MagicMock(spec=PipelineConfig)
    config.tts_engine = "google"
    config.google_tts_credentials = "fake_creds.json"
    config.google_tts_settings = {
        "voice_name": "he-IL-Neural2-M",
        "language_code": "he-IL",
        "speaking_rate": 1.0,
        "pitch": 0.0
    }
    
    engine = create_tts_engine(config)
    assert isinstance(engine, GoogleCloudTTS)
    assert engine.voice_name == "he-IL-Neural2-M"

@patch("pipeline.tts.OpenAI")
def test_create_tts_engine_openai(mock_openai):
    config = MagicMock(spec=PipelineConfig)
    config.tts_engine = "openai"
    config.openai_api_key = "fake-key"
    config.openai_tts_settings = {
        "voice": "alloy",
        "model": "tts-1",
        "speed": 1.0
    }
    config.ssl_verify = True
    
    engine = create_tts_engine(config)
    assert isinstance(engine, OpenAITTS)
    assert engine.voice == "alloy"

def test_create_tts_engine_invalid():
    config = MagicMock(spec=PipelineConfig)
    config.tts_engine = "invalid_tts"
    with pytest.raises(ValueError, match="Unsupported TTS engine"):
        create_tts_engine(config)
