"""Gemini TTS engine (gemini-3.8-flash-tts via the Gemini API Interactions endpoint) and its tts.gemini config."""
import base64
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from pipeline.config import PipelineConfig
from pipeline.factory import create_tts_engine
from pipeline.tts import GeminiTTS
from tests.section_yaml import SECTIONS_YAML

GAPS = dict(chunk_gap_paragraph_ms=700, chunk_gap_sentence_ms=300)
GEMINI_TTS = ('  engine: "gemini"\n  gemini:\n    model: "gemini-3.8-flash-tts"\n    voice: "voice_x"\n'
              '    language: "he-IL"\n    style: "warm, upbeat, measured pace"\n    chunk_size: 1000\n')


def _load(tts_yaml: str) -> PipelineConfig:
    yaml_content = ('halachic_section: "Yoreh De\'ah"\nsefaria:\n  base_url: "u"\n  timeout_seconds: 15\n  retries: 3\n  ssl_verify: true\ndirectories:\n  thinking_dir: "./output/thinking"\ngenerator:\n  engine: "gemini"\n'
                    'tts:\n  chunk_gap_paragraph_ms: 0\n  chunk_gap_sentence_ms: 0\n' + tts_yaml)
    with patch("builtins.open", mock_open(read_data=SECTIONS_YAML + yaml_content)), \
         patch("os.path.exists", return_value=True), patch("os.makedirs"):
        return PipelineConfig("config.yaml")


def _engine(chunk_size=1000, **gaps):
    return GeminiTTS("k", "gemini-3.8-flash-tts", "voice_x", "he-IL", "warm, upbeat, measured pace", True,
                     chunk_size=chunk_size, **(gaps or GAPS))


def _interaction(pcm: bytes):
    return SimpleNamespace(status="completed", output_audio=SimpleNamespace(data=base64.b64encode(pcm).decode()))


def _run(engine, text, tmp_path, responses):
    client = MagicMock()
    client.interactions.create.side_effect = responses
    out = tmp_path / "out" / "a.mp3"
    with patch("pipeline.generator.build_gemini_client", return_value=client) as build, \
         patch.object(GeminiTTS, "_assemble_pcm", wraps=engine._assemble_pcm) as assemble:
        engine.synthesize(text, str(out))
    return client, build, assemble, out


# --- config ---------------------------------------------------------------------------

def test_config_reads_gemini_tts_settings():
    cfg = _load(GEMINI_TTS)
    assert (cfg.gemini_tts_model, cfg.gemini_tts_voice, cfg.gemini_tts_language, cfg.gemini_tts_style,
            cfg.tts_chunk_size) == ("gemini-3.8-flash-tts", "voice_x", "he-IL", "warm, upbeat, measured pace", 1000)


@pytest.mark.parametrize("key", ["model", "voice", "language", "style", "chunk_size"])
def test_config_gemini_tts_settings_required(key):
    lines = [line for line in GEMINI_TTS.splitlines(keepends=True) if not line.strip().startswith(f"{key}:")]
    with pytest.raises(ValueError, match=f"'{key}' under 'tts.gemini'"):
        _load("".join(lines))


@pytest.mark.parametrize("key, value", [("model", '"gemini-3.8-flash-tts"'), ("voice", '"voice_x"'),
                                        ("language", '"he-IL"'), ("style", '"warm, upbeat, measured pace"')])
def test_config_gemini_tts_settings_must_be_non_empty_strings(key, value):
    with pytest.raises(ValueError, match=f"'tts.gemini.{key}' must be a non-empty string"):
        _load(GEMINI_TTS.replace(f"{key}: {value}", f'{key}: ""'))


def test_config_missing_gemini_section_errors():
    with pytest.raises(ValueError, match="'chunk_size' under 'tts.gemini'"):
        _load('  engine: "gemini"\n')


def test_repo_config_selects_gemini_38_tts():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        tts = yaml.safe_load(f)["tts"]
    assert tts["engine"] == "gemini"
    assert tts["gemini"] == {
        "model": "gemini-3.8-flash-tts",
        "voice": "voice_0fdvd4fjjg9o",
        "language": "he-IL",
        "style": "warm, upbeat, measured pace",
        "chunk_size": 1000,
    }


def test_factory_builds_gemini_engine_from_config():
    config = MagicMock(spec=PipelineConfig)
    config.tts_engine = "gemini"
    config.gemini_api_key = "k"
    config.gemini_tts_model, config.gemini_tts_voice = "gemini-3.8-flash-tts", "voice_x"
    config.gemini_tts_language, config.gemini_tts_style = "he-IL", "warm"
    config.ssl_verify = False
    config.tts_chunk_size = 1000
    config.tts_chunk_gap_paragraph_ms, config.tts_chunk_gap_sentence_ms = 700, 300
    engine = create_tts_engine(config)
    assert isinstance(engine, GeminiTTS)
    assert (engine.api_key, engine.model, engine.voice, engine.language, engine.style, engine.ssl_verify,
            engine._chunk_size) == ("k", "gemini-3.8-flash-tts", "voice_x", "he-IL", "warm", False, 1000)


# --- requests ---------------------------------------------------------------------------

def test_request_shape(tmp_path):
    engine = _engine()
    client, build, _, out = _run(engine, "שָׁלוֹם עֲלֵיכֶם.", tmp_path, [_interaction(b"\x01\x00" * 100)])
    build.assert_called_once_with("k", True)
    client.interactions.create.assert_called_once_with(
        model="gemini-3.8-flash-tts",
        input=[{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": "שָׁלוֹם עֲלֵיכֶם.",
                "annotations": [{"type": "speech_metadata", "style": "warm, upbeat, measured pace"}],
            }],
        }],
        response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
        generation_config={"speech_config": [{"voice": "voice_x", "language": "he-IL"}]},
    )
    assert out.exists() and out.stat().st_size > 0


def test_chunks_at_chunk_size_with_gaps_between(tmp_path):
    paras = [("א" * 600) + "." for _ in range(3)]  # 601 chars each: one paragraph per 1000-char chunk
    engine = _engine()
    pcm = [bytes([i + 1]) * 4 for i in range(3)]
    client, _, assemble, _ = _run(engine, "\n\n".join(paras), tmp_path, [_interaction(p) for p in pcm])
    texts = [c.kwargs["input"][0]["content"][0]["text"] for c in client.interactions.create.call_args_list]
    assert texts == paras
    assert all(len(t) <= 1000 for t in texts)
    chunks, gaps = assemble.call_args.args
    assert chunks == pcm
    assert gaps == [700, 700]  # paragraph boundaries


def test_long_paragraph_split_mid_paragraph_uses_sentence_gap(tmp_path):
    sentences = [("ב" * 599) + "." for _ in range(2)]
    engine = _engine()
    client, _, assemble, _ = _run(engine, " ".join(sentences), tmp_path,
                                  [_interaction(b"\x00\x00"), _interaction(b"\x00\x00")])
    assert client.interactions.create.call_count == 2
    assert assemble.call_args.args[1] == [300]


def test_pcm_concatenated_with_silence(tmp_path):
    engine = _engine()
    out = engine._assemble_pcm([b"\x01\x01", b"\x02\x02"], [700])
    assert out == b"\x01\x01" + bytes(33600) + b"\x02\x02"


def test_no_audio_raises(tmp_path):
    engine = _engine()
    empty = SimpleNamespace(status="failed", output_audio=None)
    with pytest.raises(ValueError, match="No audio returned by Gemini TTS"):
        _run(engine, "שָׁלוֹם", tmp_path, [empty])


def test_missing_api_key_raises_before_any_request(tmp_path):
    engine = GeminiTTS(None, "m", "v", "he-IL", "s", True, chunk_size=1000, **GAPS)
    with patch("pipeline.generator.build_gemini_client") as build, \
         pytest.raises(ValueError, match="GEMINI_API_KEY"):
        engine.synthesize("שָׁלוֹם", str(tmp_path / "a.mp3"))
    build.assert_not_called()
