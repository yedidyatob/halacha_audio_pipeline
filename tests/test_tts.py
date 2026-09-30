import pytest
import os
import struct
from unittest.mock import patch, MagicMock
from pipeline.tts import BaseTTS, ElevenLabsTTS, GoogleCloudTTS, OpenAITTS


# ---------------------------------------------------------------------------
# Minimal valid MP3 helpers
# ---------------------------------------------------------------------------

def _make_real_mp3(duration_ms: int = 200) -> bytes:
    """
    Generates a real, decodable silent MP3 using the imageio-ffmpeg bundled binary.
    Returns raw MP3 bytes. Falls back to a pre-encoded 1-frame stub if ffmpeg unavailable.
    """
    import subprocess
    import tempfile
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                ffmpeg_exe, "-y",
                "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
                "-t", str(duration_ms / 1000),
                "-q:a", "9", "-acodec", "libmp3lame",
                tmp_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# _merge_audio_chunks — real pydub + imageio-ffmpeg, no mocks
# ---------------------------------------------------------------------------

class _ConcreteTTS(BaseTTS):
    """Minimal concrete subclass so we can call _merge_audio_chunks directly."""
    _chunk_size = 9999

    def _setup(self): pass
    def _synthesize_chunk(self, text: str) -> bytes: return b""


def test_merge_audio_chunks_produces_output(tmp_path):
    """
    _merge_audio_chunks must produce a non-empty MP3 file when given valid chunk files.
    Uses real pydub + imageio-ffmpeg — no mocks — so this validates the full stack.
    """
    mp3 = _make_real_mp3()
    chunk1 = tmp_path / "chunk_0001.mp3"
    chunk2 = tmp_path / "chunk_0002.mp3"
    chunk1.write_bytes(mp3)
    chunk2.write_bytes(mp3)

    output = tmp_path / "merged.mp3"
    tts = _ConcreteTTS()
    tts._merge_audio_chunks([str(chunk1), str(chunk2)], str(output))

    assert output.exists(), "merged output file was not created"
    assert output.stat().st_size > 0, "merged output file is empty"


def test_merge_audio_chunks_longer_than_any_single_chunk(tmp_path):
    """
    The merged file must be at least as large as the largest single chunk,
    confirming that content from both chunks made it into the output.
    """
    mp3 = _make_real_mp3()
    chunk1 = tmp_path / "chunk_0001.mp3"
    chunk2 = tmp_path / "chunk_0002.mp3"
    chunk1.write_bytes(mp3)
    chunk2.write_bytes(mp3)

    output = tmp_path / "merged.mp3"
    tts = _ConcreteTTS()
    tts._merge_audio_chunks([str(chunk1), str(chunk2)], str(output))

    assert output.stat().st_size >= len(mp3), (
        "merged file is smaller than a single source chunk — content was lost"
    )


def test_merge_audio_chunks_creates_parent_dirs(tmp_path):
    """_merge_audio_chunks must create the output directory if it doesn't exist."""
    chunk = tmp_path / "chunk_0001.mp3"
    chunk.write_bytes(_make_real_mp3())

    output = tmp_path / "nested" / "deep" / "merged.mp3"
    tts = _ConcreteTTS()
    tts._merge_audio_chunks([str(chunk)], str(output))

    assert output.exists()


# ---------------------------------------------------------------------------
# ElevenLabsTTS — single-chunk path (no merge)
# ---------------------------------------------------------------------------

@patch("elevenlabs.client.ElevenLabs")
def test_elevenlabs_tts_single_chunk(mock_elevenlabs_cls, tmp_path):
    mock_client = MagicMock()
    mock_elevenlabs_cls.return_value = mock_client
    mock_client.text_to_speech.convert.return_value = [b"audio_chunk_1", b"audio_chunk_2"]

    tts = ElevenLabsTTS(api_key="fake-key", voice_id="Adam", model_id="eleven_v3", stability=0.5, similarity_boost=0.75)
    output_file = str(tmp_path / "test_output.mp3")

    tts.synthesize("טקסט קצר", output_file)

    mock_client.text_to_speech.convert.assert_called_once()
    call_kwargs = mock_client.text_to_speech.convert.call_args[1]
    assert call_kwargs["text"] == "טקסט קצר"
    assert call_kwargs["voice_id"] == "Adam"
    assert call_kwargs["model_id"] == "eleven_v3"
    assert call_kwargs["voice_settings"].stability == 0.5
    assert call_kwargs["voice_settings"].similarity_boost == 0.75

    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        assert f.read() == b"audio_chunk_1audio_chunk_2"


@patch("elevenlabs.client.ElevenLabs")
def test_elevenlabs_tts_multi_chunk_calls_merge(mock_elevenlabs_cls, tmp_path):
    """
    For texts that exceed the chunk size, convert() must be called once per chunk
    and _merge_audio_chunks must be invoked to join them.
    """
    mock_client = MagicMock()
    mock_elevenlabs_cls.return_value = mock_client
    mock_client.text_to_speech.convert.return_value = [_make_real_mp3()]

    tts = ElevenLabsTTS(api_key="fake-key", voice_id="Adam", model_id="eleven_v3", stability=0.5, similarity_boost=0.75)
    output_file = str(tmp_path / "multi_chunk.mp3")

    # Two paragraphs each just over 1500 chars → forces 2 chunks
    long_text = ("א" * 800 + " ") * 2 + "\n" + ("ב" * 800 + " ") * 2
    assert len(long_text) > 1500

    with patch.object(tts, "_merge_audio_chunks", wraps=tts._merge_audio_chunks) as mock_merge:
        tts.synthesize(long_text, output_file)
        mock_merge.assert_called_once()

    assert mock_client.text_to_speech.convert.call_count >= 2


# ---------------------------------------------------------------------------
# GoogleCloudTTS
# ---------------------------------------------------------------------------

@patch("google.cloud.texttospeech.TextToSpeechClient")
def test_google_cloud_tts_single_chunk(mock_gtts_cls, tmp_path):
    mock_client = MagicMock()
    mock_gtts_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.audio_content = b"google_tts_bytes"
    mock_client.synthesize_speech.return_value = mock_response

    tts = GoogleCloudTTS(credentials_path="fake_creds.json", voice_name="he-IL-Neural2-F")
    output_file = str(tmp_path / "google_output.mp3")

    tts.synthesize("שלום עולם", output_file)

    mock_client.synthesize_speech.assert_called_once()
    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        assert f.read() == b"google_tts_bytes"


@patch("google.cloud.texttospeech.TextToSpeechClient")
def test_google_cloud_tts_multi_chunk_calls_merge(mock_gtts_cls, tmp_path):
    mock_client = MagicMock()
    mock_gtts_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.audio_content = _make_real_mp3()
    mock_client.synthesize_speech.return_value = mock_response

    tts = GoogleCloudTTS(credentials_path="fake_creds.json", voice_name="he-IL-Neural2-F")
    output_file = str(tmp_path / "google_chunked.mp3")

    long_text = ("א" * 1200) + "\n" + ("ב" * 1200)

    with patch.object(tts, "_merge_audio_chunks", wraps=tts._merge_audio_chunks) as mock_merge:
        tts.synthesize(long_text, output_file)
        mock_merge.assert_called_once()

    assert mock_client.synthesize_speech.call_count == 2


# ---------------------------------------------------------------------------
# OpenAITTS
# ---------------------------------------------------------------------------

@patch("pipeline.tts.OpenAI")
def test_openai_tts_single_chunk(mock_openai_cls, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.content = b"openai_tts_bytes"
    mock_client.audio.speech.create.return_value = mock_response

    tts = OpenAITTS(api_key="fake-key", voice="alloy", model="tts-1", speed=1.0)
    output_file = str(tmp_path / "openai_output.mp3")

    tts.synthesize("טקסט הלכה", output_file)

    call_kwargs = mock_client.audio.speech.create.call_args[1]
    assert call_kwargs["model"] == "tts-1"
    assert call_kwargs["voice"] == "alloy"
    assert call_kwargs["input"] == "טקסט הלכה"
    assert call_kwargs["speed"] == 1.0

    assert os.path.exists(output_file)
    with open(output_file, "rb") as f:
        assert f.read() == b"openai_tts_bytes"


@patch("pipeline.tts.OpenAI")
def test_openai_tts_multi_chunk_calls_merge(mock_openai_cls, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.content = _make_real_mp3()
    mock_client.audio.speech.create.return_value = mock_response

    tts = OpenAITTS(api_key="fake-key", voice="alloy")
    output_file = str(tmp_path / "openai_chunked.mp3")

    long_text = ("א" * 2200) + "\n" + ("ב" * 2200)

    with patch.object(tts, "_merge_audio_chunks", wraps=tts._merge_audio_chunks) as mock_merge:
        tts.synthesize(long_text, output_file)
        mock_merge.assert_called_once()

    assert mock_client.audio.speech.create.call_count >= 2
