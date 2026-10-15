"""TTS chunk_size config and ElevenLabs previous_text / next_text request stitching."""
import os
from unittest.mock import MagicMock, patch, mock_open

import pytest

from pipeline.config import PipelineConfig
from pipeline.factory import create_tts_engine
from pipeline.tts import (
    ElevenLabsTTS, GoogleCloudTTS, OpenAITTS, GeminiTTS,
    context_head, context_tail, strip_phoneme_markup, validate_chunk_size,
)

GAPS = dict(chunk_gap_paragraph_ms=0, chunk_gap_sentence_ms=0)
TAG = '<phoneme alphabet="ipa" ph="tet">טית</phoneme>'


def _load(tts_yaml: str) -> PipelineConfig:
    yaml_content = ('halachic_section: "Yoreh De\'ah"\ngenerator:\n  engine: "gemini"\n'
                    'tts:\n  chunk_gap_paragraph_ms: 0\n  chunk_gap_sentence_ms: 0\n' + tts_yaml)
    with patch("builtins.open", mock_open(read_data=yaml_content)), \
         patch("os.path.exists", return_value=True), patch("os.makedirs"):
        return PipelineConfig("config.yaml")


# --- config validation --------------------------------------------------------

@pytest.mark.parametrize("engine,size", [("elevenlabs", 8000), ("google", 2000), ("openai", 4000), ("gemini", 2000)])
def test_config_reads_chunk_size_for_selected_engine(engine, size):
    extra = "    context_chars: 500\n" if engine == "elevenlabs" else ""
    gem = '    model_name: "m"\n    voice_name: "Achird"\n' if engine == "gemini" else ""
    cfg = _load(f'  engine: "{engine}"\n  {engine}:\n    chunk_size: {size}\n' + extra + gem)
    assert cfg.tts_chunk_size == size


def test_config_missing_chunk_size_errors():
    with pytest.raises(ValueError, match="'chunk_size' under 'tts.openai'"):
        _load('  engine: "openai"\n  openai:\n    voice: "alloy"\n')


@pytest.mark.parametrize("bad", ["0", "-5", "1.5", '"2000"', "true"])
def test_config_invalid_chunk_size_errors(bad):
    with pytest.raises(ValueError, match="positive integer"):
        _load(f'  engine: "openai"\n  openai:\n    chunk_size: {bad}\n')


def test_config_google_chunk_size_over_byte_limit_errors():
    assert _load('  engine: "google"\n  google:\n    chunk_size: 2500\n').tts_chunk_size == 2500
    with pytest.raises(ValueError, match="5000 bytes"):
        _load('  engine: "google"\n  google:\n    chunk_size: 2501\n')


def test_config_elevenlabs_context_chars_required_and_validated():
    with pytest.raises(ValueError, match="context_chars"):
        _load('  engine: "elevenlabs"\n  elevenlabs:\n    chunk_size: 8000\n')
    with pytest.raises(ValueError, match="positive integer"):
        _load('  engine: "elevenlabs"\n  elevenlabs:\n    chunk_size: 8000\n    context_chars: 0\n')
    cfg = _load('  engine: "elevenlabs"\n  elevenlabs:\n    chunk_size: 8000\n    context_chars: 300\n')
    assert cfg.elevenlabs_context_chars == 300


def test_repo_config_yaml_chunk_settings():
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        tts = yaml.safe_load(f)["tts"]
    assert tts["elevenlabs"]["chunk_size"] == 8000
    assert tts["elevenlabs"]["context_chars"] == 500
    assert tts["google"]["chunk_size"] == 2000
    assert tts["openai"]["chunk_size"] == 4000
    assert tts["gemini"]["chunk_size"] == 2000


def test_engines_store_and_validate_chunk_size():
    assert OpenAITTS(api_key="k", chunk_size=4000, **GAPS)._chunk_size == 4000
    assert GoogleCloudTTS(chunk_size=2000, **GAPS)._chunk_size == 2000
    with patch("google.auth.default", side_effect=Exception("no creds")):
        assert GeminiTTS("m", "v", chunk_size=1234, **GAPS)._chunk_size == 1234
    el = ElevenLabsTTS(api_key="k", chunk_size=8000, context_chars=500, **GAPS)
    assert (el._chunk_size, el.context_chars) == (8000, 500)
    with pytest.raises(ValueError):
        OpenAITTS(api_key="k", chunk_size=0, **GAPS)
    with pytest.raises(ValueError):
        ElevenLabsTTS(api_key="k", chunk_size=8000, context_chars=-1, **GAPS)
    with pytest.raises(ValueError, match="5000 bytes"):
        GoogleCloudTTS(chunk_size=3000, **GAPS)
    with pytest.raises(TypeError):
        OpenAITTS(api_key="k", **GAPS)  # no code default for chunk_size
    for bad in (None, True, 2.0, "10"):
        with pytest.raises(ValueError):
            validate_chunk_size(bad)


def test_google_rejects_chunk_over_byte_limit():
    g = GoogleCloudTTS(chunk_size=2500, **GAPS)
    g._texttospeech = MagicMock()
    g._google_client = MagicMock()
    with pytest.raises(ValueError, match="bytes"):
        g._synthesize_chunk("ש" * 2501)  # 5002 bytes
    g._google_client.synthesize_speech.assert_not_called()


def test_factory_passes_chunk_settings():
    config = MagicMock(spec=PipelineConfig)
    config.tts_engine = "elevenlabs"
    config.tts_chunk_size = 7000
    config.elevenlabs_context_chars = 321
    config.tts_chunk_gap_paragraph_ms = 0
    config.tts_chunk_gap_sentence_ms = 0
    config.elevenlabs_api_key = "k"
    config.elevenlabs_settings = {"voice_id": "v", "model_id": "eleven_v4", "stability": 0.5, "similarity_boost": 0.75}
    config.ssl_verify = True
    engine = create_tts_engine(config)
    assert (engine._chunk_size, engine.context_chars) == (7000, 321)


# --- context helpers ------------------------------------------------------------

def test_context_tail_and_head_cut_at_word_boundary():
    text = "alpha beta gamma delta"
    assert context_tail(text, 100) == text
    assert context_tail(text, 9) == "delta"        # "ma delta" -> partial "ma" dropped
    assert context_tail(text, 11) == "gamma delta"  # exact boundary kept
    assert context_head(text, 9) == "alpha"        # "alpha bet" -> partial "bet" dropped
    assert context_head(text, 10) == "alpha beta"
    assert context_tail("abcdefghij", 5) == ""      # no boundary fits
    assert context_head("abcdefghij", 5) == ""


def test_context_strips_phoneme_tags():
    text = f"סימן {TAG} עוסק בדין"
    assert strip_phoneme_markup(text) == "סימן טית עוסק בדין"
    for n in range(1, 40):
        for out in (context_tail(text, n), context_head(text, n)):
            assert "<" not in out and ">" not in out and len(out) <= n


# --- ElevenLabs stitching ------------------------------------------------------

def _el(chunk_size, context_chars):
    tts = ElevenLabsTTS(api_key="k", voice_id="v", model_id="eleven_v4", stability=0.5,
                        similarity_boost=0.75, chunk_size=chunk_size, context_chars=context_chars, **GAPS)
    return tts


def _run(tts, text, tmp_path):
    with patch("elevenlabs.client.ElevenLabs") as cls:
        client = cls.return_value
        client.text_to_speech.convert.return_value = [b"mp3"]
        with patch.object(tts, "_merge_audio_chunks"):
            tts.synthesize(text, str(tmp_path / "out.mp3"))
    return [c.kwargs for c in client.text_to_speech.convert.call_args_list]


def test_single_chunk_sends_no_context(tmp_path):
    calls = _run(_el(8000, 500), "טקסט קצר עם תגית " + TAG, tmp_path)
    assert len(calls) == 1
    assert "previous_text" not in calls[0] and "next_text" not in calls[0]


def test_previous_and_next_text_per_chunk(tmp_path):
    paras = [f"פסקה {i} " + " ".join(f"מילה{i}_{j}" for j in range(8)) for i in range(3)]
    tts = _el(len(paras[0]) + 5, 20)
    calls = _run(tts, "\n".join(paras), tmp_path)
    assert [c["text"] for c in calls] == paras
    assert "previous_text" not in calls[0]
    assert "next_text" not in calls[-1]
    for i, c in enumerate(calls):
        if i > 0:
            assert c["previous_text"] == context_tail(paras[i - 1], 20)
            assert paras[i - 1].endswith(c["previous_text"]) and 0 < len(c["previous_text"]) <= 20
        if i < len(calls) - 1:
            assert c["next_text"] == context_head(paras[i + 1], 20)
            assert paras[i + 1].startswith(c["next_text"]) and 0 < len(c["next_text"]) <= 20


def test_tags_never_split_in_chunks_or_context(tmp_path):
    words = []
    for i in range(60):
        words.append(TAG if i % 3 == 0 else f"מילה{i}")
    text = " ".join(words) + "."
    calls = _run(_el(120, 40), text, tmp_path)
    assert len(calls) > 2
    for c in calls:
        assert c["text"].count("<phoneme") == c["text"].count("</phoneme>")
        for key in ("previous_text", "next_text"):
            if key in c:
                assert "<" not in c[key] and ">" not in c[key] and "phoneme" not in c[key]
