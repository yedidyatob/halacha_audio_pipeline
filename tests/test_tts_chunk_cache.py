"""Saved TTS chunks (reused on a rerun, keyed by text / context / settings, written atomically) and the
configurable TTS request timeout. No network and no real audio: synthesis calls are mocked."""
import base64
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from pipeline.config import PipelineConfig
from pipeline.factory import create_tts_engine
from pipeline.tts import BaseTTS, ElevenLabsTTS, GeminiTTS, OpenAITTS, _write_atomic
from tests.section_yaml import SECTIONS_YAML

GAPS = dict(chunk_gap_paragraph_ms=0, chunk_gap_sentence_ms=0)
PARAGRAPHS = ["א" * 60, "ב" * 60, "ג" * 60]  # one paragraph per chunk at chunk_size 80
TEXT = "\n".join(PARAGRAPHS)


class _FakeTTS(BaseTTS):
    """Records every synthesized chunk; the 'audio' is the chunk text, tagged with the voice."""

    def __init__(self, voice="v1", keep_chunks=True, fail_on=None):
        self._set_chunk_gaps(0, 0)
        self._set_chunk_size(80)
        self._set_keep_chunks(keep_chunks)
        self.voice = voice
        self.fail_on = fail_on
        self.calls = []

    def _cache_settings(self):
        return {"voice": self.voice}

    def _synthesize_chunk(self, text):
        if text == self.fail_on:
            raise TimeoutError("read timeout")
        self.calls.append(text)
        return f"{self.voice}:{text}".encode("utf-8")


def _merge_by_concat(chunk_paths, output_path, gaps_ms=None):
    with open(output_path, "wb") as out:
        for path in chunk_paths:
            with open(path, "rb") as f:
                out.write(f.read())


def _synthesize(tts, tmp_path, text=TEXT):
    chunk_dir = tmp_path / "chunks" / "lesson"
    with patch.object(tts, "_merge_audio_chunks", side_effect=_merge_by_concat):
        tts.synthesize(text, str(tmp_path / "out" / "lesson.mp3"), str(chunk_dir))
    return chunk_dir


def _chunk_files(chunk_dir):
    return sorted(os.listdir(chunk_dir))


# --- saving and reuse ------------------------------------------------------------

def test_each_chunk_is_saved_after_synthesis(tmp_path):
    tts = _FakeTTS()
    chunk_dir = _synthesize(tts, tmp_path)
    files = _chunk_files(chunk_dir)
    assert [name.split("_")[1] for name in files] == ["0001", "0002", "0003"]
    assert all(name.endswith(".mp3") for name in files)
    assert [(chunk_dir / name).read_bytes() for name in files] == [f"v1:{p}".encode() for p in PARAGRAPHS]
    assert (tmp_path / "out" / "lesson.mp3").read_bytes() == "".join(f"v1:{p}" for p in PARAGRAPHS).encode()


def test_chunks_saved_before_a_later_chunk_fails(tmp_path):
    tts = _FakeTTS(fail_on=PARAGRAPHS[1])
    with pytest.raises(TimeoutError):
        _synthesize(tts, tmp_path)
    files = _chunk_files(tmp_path / "chunks" / "lesson")
    assert len(files) == 1 and files[0].startswith("chunk_0001_")  # the paid-for first chunk survives


def test_rerun_synthesizes_only_missing_chunks(tmp_path):
    with pytest.raises(TimeoutError):
        _synthesize(_FakeTTS(fail_on=PARAGRAPHS[1]), tmp_path)
    rerun = _FakeTTS()
    _synthesize(rerun, tmp_path)
    assert rerun.calls == PARAGRAPHS[1:]
    assert (tmp_path / "out" / "lesson.mp3").read_bytes() == "".join(f"v1:{p}" for p in PARAGRAPHS).encode()


def test_full_rerun_with_kept_chunks_synthesizes_nothing(tmp_path):
    _synthesize(_FakeTTS(), tmp_path)
    rerun = _FakeTTS()
    _synthesize(rerun, tmp_path)
    assert rerun.calls == []


def test_changed_text_invalidates_only_its_chunk(tmp_path):
    _synthesize(_FakeTTS(), tmp_path)
    edited = [PARAGRAPHS[0], "ד" * 60, PARAGRAPHS[2]]
    rerun = _FakeTTS()
    _synthesize(rerun, tmp_path, text="\n".join(edited))
    assert rerun.calls == ["ד" * 60]
    assert len(_chunk_files(tmp_path / "chunks" / "lesson")) == 3  # the stale chunk was removed


def test_changed_voice_invalidates_every_chunk(tmp_path):
    _synthesize(_FakeTTS(voice="v1"), tmp_path)
    rerun = _FakeTTS(voice="v2")
    _synthesize(rerun, tmp_path)
    assert rerun.calls == PARAGRAPHS
    assert (tmp_path / "out" / "lesson.mp3").read_bytes() == "".join(f"v2:{p}" for p in PARAGRAPHS).encode()


def test_changed_chunking_invalidates_the_cache(tmp_path):
    _synthesize(_FakeTTS(), tmp_path)
    rerun = _FakeTTS()
    rerun._set_chunk_size(200)  # the whole text in one chunk
    _synthesize(rerun, tmp_path)
    assert rerun.calls == [TEXT]


def _elevenlabs(voice_id="voice_a"):
    return ElevenLabsTTS(api_key="k", voice_id=voice_id, model_id="eleven_v4", stability=0.5, similarity_boost=0.75,
                         chunk_size=80, context_chars=20, language_code="he", keep_chunks=True,
                         request_timeout_seconds=1800, **GAPS)


def _elevenlabs_run(tts, tmp_path, text=TEXT):
    with patch("elevenlabs.client.ElevenLabs") as cls:
        convert = cls.return_value.text_to_speech.convert
        convert.side_effect = lambda **kw: [kw["text"].encode("utf-8")]
        _synthesize(tts, tmp_path, text)
    return [c.kwargs["text"] for c in convert.call_args_list]


def test_elevenlabs_voice_change_invalidates_cache(tmp_path):
    assert _elevenlabs_run(_elevenlabs(), tmp_path) == PARAGRAPHS
    assert _elevenlabs_run(_elevenlabs(), tmp_path) == []
    assert _elevenlabs_run(_elevenlabs(voice_id="voice_b"), tmp_path) == PARAGRAPHS


def test_elevenlabs_neighbour_edit_invalidates_its_stitched_neighbours(tmp_path):
    worded = [" ".join([letter * 5] * 10) for letter in "אבג"]  # 59 chars each, so context cuts at words
    _elevenlabs_run(_elevenlabs(), tmp_path, "\n".join(worded))
    edited = worded[:2] + [" ".join(["דדדדד"] * 10)]
    # Chunk 2's next_text changes with chunk 3, so both are synthesized again; chunk 1 is reused.
    assert _elevenlabs_run(_elevenlabs(), tmp_path, "\n".join(edited)) == edited[1:]


def test_gemini_chunks_are_saved_as_pcm_and_reused(tmp_path):
    engine = GeminiTTS("k", "m", "v", "he-IL", "s", True, chunk_size=80, keep_chunks=True, **GAPS)
    client = MagicMock()
    client.interactions.create.side_effect = lambda **kw: SimpleNamespace(
        status="completed", output_audio=SimpleNamespace(data=base64.b64encode(b"\x01\x00").decode()))
    with patch("pipeline.generator.build_gemini_client", return_value=client):
        engine.synthesize(TEXT, str(tmp_path / "a.mp3"), str(tmp_path / "chunks"))
        engine.synthesize(TEXT, str(tmp_path / "b.mp3"), str(tmp_path / "chunks"))
    assert client.interactions.create.call_count == 3  # second run reused every chunk
    files = _chunk_files(tmp_path / "chunks")
    assert len(files) == 3 and all(name.endswith(".pcm") for name in files)


# --- atomic writes ---------------------------------------------------------------

def test_leftover_partial_file_is_not_reused(tmp_path):
    _synthesize(_FakeTTS(), tmp_path)
    chunk_dir = tmp_path / "chunks" / "lesson"
    first = chunk_dir / _chunk_files(chunk_dir)[0]
    partial = first.read_bytes()[:3]
    first.unlink()
    (chunk_dir / (first.name + ".part")).write_bytes(partial)  # an interrupted earlier write
    rerun = _FakeTTS()
    _synthesize(rerun, tmp_path)
    assert rerun.calls == [PARAGRAPHS[0]]
    assert first.read_bytes() == f"v1:{PARAGRAPHS[0]}".encode()
    assert not (chunk_dir / (first.name + ".part")).exists()  # stale files are cleaned after the merge


def test_interrupted_write_leaves_no_chunk(tmp_path):
    target = tmp_path / "chunk_0001_abc.mp3"
    with patch("pipeline.tts.os.replace", side_effect=OSError("disk full")), pytest.raises(OSError):
        _write_atomic(str(target), b"audio")
    assert os.listdir(tmp_path) == []


# --- cleanup after a successful merge --------------------------------------------

def test_keep_chunks_false_deletes_the_lesson_folder(tmp_path):
    chunk_dir = _synthesize(_FakeTTS(keep_chunks=False), tmp_path)
    assert not chunk_dir.exists()
    assert (tmp_path / "out" / "lesson.mp3").exists()


def test_failed_run_keeps_chunks_even_with_keep_chunks_false(tmp_path):
    with pytest.raises(TimeoutError):
        _synthesize(_FakeTTS(keep_chunks=False, fail_on=PARAGRAPHS[2]), tmp_path)
    assert len(_chunk_files(tmp_path / "chunks" / "lesson")) == 2


def test_single_chunk_is_copied_to_the_output(tmp_path):
    tts = _FakeTTS()
    _synthesize(tts, tmp_path, text="קצר")
    assert (tmp_path / "out" / "lesson.mp3").read_bytes() == "v1:קצר".encode()


@pytest.mark.parametrize("bad", [None, "yes", 1])
def test_keep_chunks_must_be_bool(bad):
    with pytest.raises(ValueError, match="keep_chunks"):
        _FakeTTS(keep_chunks=bad)


# --- request timeout -------------------------------------------------------------

def _load(tts_extra: str) -> PipelineConfig:
    yaml_content = ('halachic_section: "Yoreh De\'ah"\nsefaria:\n  base_url: "u"\n  timeout_seconds: 15\n  retries: 3\n'
                    '  ssl_verify: true\ndirectories:\n  thinking_dir: "./output/thinking"\n'
                    '  tts_chunks_dir: "./output/tts_chunks"\ngenerator:\n  engine: "gemini"\n'
                    'tts:\n  engine: "openai"\n  chunk_gap_paragraph_ms: 0\n  chunk_gap_sentence_ms: 0\n'
                    '  openai:\n    chunk_size: 4000\n' + tts_extra)
    with patch("builtins.open", mock_open(read_data=SECTIONS_YAML + yaml_content)), \
         patch("os.path.exists", return_value=True), patch("os.makedirs"):
        return PipelineConfig("config.yaml")


def test_config_reads_timeout_and_keep_chunks():
    cfg = _load("  keep_chunks: false\n  request_timeout_seconds: 900\n")
    assert (cfg.tts_keep_chunks, cfg.tts_request_timeout_seconds, cfg.tts_chunks_dir) == (
        False, 900.0, "./output/tts_chunks")


@pytest.mark.parametrize("extra, missing", [
    ("  request_timeout_seconds: 900\n", "keep_chunks"),
    ("  keep_chunks: true\n", "request_timeout_seconds"),
])
def test_config_requires_timeout_and_keep_chunks(extra, missing):
    with pytest.raises(ValueError, match=f"'{missing}' under 'tts'"):
        _load(extra)


@pytest.mark.parametrize("bad", [0, -5, "1800", True])
def test_config_rejects_invalid_timeout(bad):
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        _load(f"  keep_chunks: true\n  request_timeout_seconds: {yaml.safe_dump(bad).splitlines()[0]}\n")


def test_config_requires_tts_chunks_dir():
    yaml_content = ('halachic_section: "Yoreh De\'ah"\nsefaria:\n  base_url: "u"\n  timeout_seconds: 15\n  retries: 3\n'
                    '  ssl_verify: true\ndirectories:\n  thinking_dir: "./output/thinking"\ngenerator:\n  engine: "gemini"\n'
                    'tts:\n  engine: "openai"\n  chunk_gap_paragraph_ms: 0\n  chunk_gap_sentence_ms: 0\n'
                    '  keep_chunks: true\n  request_timeout_seconds: 1800\n  openai:\n    chunk_size: 4000\n')
    with patch("builtins.open", mock_open(read_data=SECTIONS_YAML + yaml_content)), \
         patch("os.path.exists", return_value=True), patch("os.makedirs"), \
         pytest.raises(ValueError, match="'tts_chunks_dir' under 'directories'"):
        PipelineConfig("config.yaml")


def _factory_config(engine):
    config = MagicMock(spec=PipelineConfig)
    config.tts_engine = engine
    config.tts_chunk_size = 4000
    config.tts_chunk_gap_paragraph_ms = 0
    config.tts_chunk_gap_sentence_ms = 0
    config.tts_keep_chunks = False
    config.tts_request_timeout_seconds = 1234.0
    config.ssl_verify = True
    config.elevenlabs_api_key = config.openai_api_key = "k"
    config.elevenlabs_context_chars = 500
    config.elevenlabs_language_code = "he"
    config.elevenlabs_settings = {"voice_id": "v", "model_id": "eleven_v4", "stability": 0.5, "similarity_boost": 0.75}
    config.openai_tts_settings = {"voice": "alloy", "model": "tts-1", "speed": 1.0}
    return config


def test_elevenlabs_timeout_comes_from_config():
    engine = create_tts_engine(_factory_config("elevenlabs"))
    assert (engine.request_timeout_seconds, engine.keep_chunks) == (1234.0, False)
    with patch("httpx.Client") as httpx_client, patch("elevenlabs.client.ElevenLabs") as client_cls:
        engine._setup()
    assert httpx_client.call_args.kwargs["timeout"] == 1234.0
    assert client_cls.call_args.kwargs["timeout"] == 1234.0


def test_openai_timeout_comes_from_config():
    engine = create_tts_engine(_factory_config("openai"))
    with patch("httpx.Client") as httpx_client, patch("pipeline.tts.OpenAI") as client_cls:
        engine._setup()
    assert httpx_client.call_args.kwargs["timeout"] == 1234.0
    assert client_cls.call_args.kwargs["timeout"] == 1234.0


def test_engines_require_timeout_argument():
    with pytest.raises(TypeError, match="request_timeout_seconds"):
        OpenAITTS(api_key="k", ssl_verify=True, speed=1.0, model="tts-1", voice="alloy", chunk_size=4000,
                  keep_chunks=True, **GAPS)


def test_repo_config_yaml_chunk_cache_and_timeout_defaults():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["tts"]["keep_chunks"] is True
    assert data["tts"]["request_timeout_seconds"] == 1800
    assert data["directories"]["tts_chunks_dir"] == "./output/tts_chunks"
