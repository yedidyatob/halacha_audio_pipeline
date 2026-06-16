import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
from pipeline.tts import ElevenLabsTTS, GoogleCloudTTS, OpenAITTS

@patch("elevenlabs.client.ElevenLabs")
def test_elevenlabs_tts_success(mock_elevenlabs_cls, tmp_path):
    mock_client = MagicMock()
    mock_elevenlabs_cls.return_value = mock_client
    
    # Mock return bytes from convert
    mock_client.text_to_speech.convert.return_value = [b"audio_chunk_1", b"audio_chunk_2"]

    tts = ElevenLabsTTS(api_key="fake-key", voice_id="Adam")
    output_file = os.path.join(tmp_path, "test_output.mp3")
    
    tts.synthesize("טקסט כלשהו", output_file)

    # Verify convert is called with voice parameters
    mock_client.text_to_speech.convert.assert_called_once()
    call_args, call_kwargs = mock_client.text_to_speech.convert.call_args
    assert call_kwargs["text"] == "טקסט כלשהו"
    assert call_kwargs["voice_id"] == "Adam"
    assert call_kwargs["model_id"] == "eleven_v3"
    assert call_kwargs["voice_settings"].stability == 0.5
    assert call_kwargs["voice_settings"].similarity_boost == 0.75
    
    # Verify file is written
    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        content = f.read()
        assert content == b"audio_chunk_1audio_chunk_2"


@patch("google.cloud.texttospeech.TextToSpeechClient")
def test_google_cloud_tts_success(mock_gtts_cls, tmp_path):
    mock_client = MagicMock()
    mock_gtts_cls.return_value = mock_client
    
    # Mock the return payload
    mock_response = MagicMock()
    mock_response.audio_content = b"google_tts_bytes"
    mock_client.synthesize_speech.return_value = mock_response

    tts = GoogleCloudTTS(credentials_path="fake_creds.json", voice_name="he-IL-Neural2-F")
    output_file = os.path.join(tmp_path, "google_output.mp3")
    
    tts.synthesize("שלום עולם", output_file)

    # Verify synthesizer calls
    mock_client.synthesize_speech.assert_called_once()
    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        content = f.read()
        assert content == b"google_tts_bytes"


@patch("google.cloud.texttospeech.TextToSpeechClient")
def test_google_cloud_tts_chunking(mock_gtts_cls, tmp_path):
    mock_client = MagicMock()
    mock_gtts_cls.return_value = mock_client
    
    # Mock the return payload for multiple calls
    mock_response_1 = MagicMock()
    mock_response_1.audio_content = b"part_1_"
    mock_response_2 = MagicMock()
    mock_response_2.audio_content = b"part_2"
    mock_client.synthesize_speech.side_effect = [mock_response_1, mock_response_2]

    tts = GoogleCloudTTS(credentials_path="fake_creds.json", voice_name="he-IL-Neural2-F")
    output_file = os.path.join(tmp_path, "google_chunked_output.mp3")
    
    # Generate a string longer than 2000 characters (e.g. 2200 characters)
    # Using a paragraph of 1200 characters, followed by a newline, followed by another 1200 characters
    long_text = ("א" * 1200) + "\n" + ("ב" * 1200)
    tts.synthesize(long_text, output_file)

    # Verify synthesizer calls: should be called twice
    assert mock_client.synthesize_speech.call_count == 2
    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        content = f.read()
        assert content == b"part_1_part_2"


@patch("pipeline.tts.OpenAI")
def test_openai_tts_success(mock_openai_cls, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.content = b"openai_tts_bytes"
    mock_client.audio.speech.create.return_value = mock_response

    tts = OpenAITTS(api_key="fake-key", voice="alloy", model="tts-1", speed=1.0)
    output_file = os.path.join(tmp_path, "openai_output.mp3")
    
    tts.synthesize("טקסט הלכה", output_file)

    mock_client.audio.speech.create.assert_called_once()
    call_kwargs = mock_client.audio.speech.create.call_args[1]
    assert call_kwargs["model"] == "tts-1"
    assert call_kwargs["voice"] == "alloy"
    assert call_kwargs["input"] == "טקסט הלכה"
    assert call_kwargs["speed"] == 1.0
    
    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        content = f.read()
        assert content == b"openai_tts_bytes"


@patch("pipeline.tts.OpenAI")
def test_openai_tts_chunking(mock_openai_cls, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_response_1 = MagicMock()
    mock_response_1.content = b"part_1_"
    mock_response_2 = MagicMock()
    mock_response_2.content = b"part_2"
    mock_client.audio.speech.create.side_effect = [mock_response_1, mock_response_2]

    tts = OpenAITTS(api_key="fake-key", voice="alloy")
    output_file = os.path.join(tmp_path, "openai_chunked_output.mp3")
    
    # Generate text longer than 4000 characters
    long_text = ("א" * 2200) + "\n" + ("ב" * 2200)
    tts.synthesize(long_text, output_file)

    assert mock_client.audio.speech.create.call_count == 2
    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        content = f.read()
        assert content == b"part_1_part_2"

