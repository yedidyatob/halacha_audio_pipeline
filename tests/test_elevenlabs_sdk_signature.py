"""Guards against kwarg-name drift: checks the REAL installed elevenlabs SDK, not a mock."""
import inspect
from unittest.mock import patch

import pytest

elevenlabs_client = pytest.importorskip("elevenlabs.client")

from pipeline.tts import ElevenLabsTTS


def _real_convert_params():
    client = elevenlabs_client.ElevenLabs(api_key="test-key")
    return inspect.signature(client.text_to_speech.convert).parameters


def test_installed_sdk_convert_accepts_stitching_and_language_code():
    params = _real_convert_params()
    for name in ("previous_text", "next_text", "language_code"):
        assert name in params, f"elevenlabs SDK too old for {name}: requires >=1.6.0 (see requirements.txt)"


def test_every_kwarg_we_send_exists_in_real_sdk_signature(tmp_path):
    tts = ElevenLabsTTS(api_key="k", voice_id="v", model_id="eleven_v4", stability=0.5,
                        similarity_boost=0.75, chunk_size=60, context_chars=20, language_code="he",
                        chunk_gap_paragraph_ms=0, chunk_gap_sentence_ms=0, keep_chunks=True,
                        request_timeout_seconds=1800)
    text = "\n".join(f"פסקה {i} " + " ".join(f"מילה{j}" for j in range(6)) for i in range(3))
    with patch("elevenlabs.client.ElevenLabs") as cls, patch.object(tts, "_merge_audio_chunks"):
        convert = cls.return_value.text_to_speech.convert
        convert.return_value = [b"mp3"]
        tts.synthesize(text, str(tmp_path / "out.mp3"), str(tmp_path / "chunks"))
    sent = set().union(*(c.kwargs.keys() for c in convert.call_args_list))
    assert {"previous_text", "next_text", "language_code"} <= sent  # the multi-chunk path really sends them
    assert sent <= set(_real_convert_params()), f"unknown kwargs: {sent - set(_real_convert_params())}"
