"""Tests for inter-chunk silence: chunk break types, gap planning, config, and merging."""
import os
import subprocess
import tempfile
from unittest.mock import patch, mock_open, MagicMock

import pytest

from pipeline.config import PipelineConfig
from pipeline.factory import create_tts_engine
from pipeline.tts import (
    BaseTTS,
    ElevenLabsTTS,
    GeminiTTS,
    GoogleCloudTTS,
    OpenAITTS,
    plan_gaps,
    validate_gap_ms,
    BREAK_PARAGRAPH,
    BREAK_SENTENCE,
)


class _ConcreteTTS(BaseTTS):
    _chunk_size = 9999

    def _setup(self):
        pass

    def _synthesize_chunk(self, text: str) -> bytes:
        return b""


# ---------------------------------------------------------------------------
# Chunker break types
# ---------------------------------------------------------------------------

def test_single_chunk_short_text():
    tts = _ConcreteTTS()
    assert tts._chunk_text_with_breaks("hello\nworld", max_chars=100) == [
        ("hello\nworld", BREAK_PARAGRAPH)
    ]
    assert tts._chunk_text("hello\nworld", max_chars=100) == ["hello\nworld"]


def test_paragraph_boundaries():
    tts = _ConcreteTTS()
    text = "\n".join(["a" * 40, "b" * 40, "c" * 40])
    result = tts._chunk_text_with_breaks(text, max_chars=90)
    assert [c for c, _ in result] == ["a" * 40 + "\n\n" + "b" * 40, "c" * 40]
    assert all(bt == BREAK_PARAGRAPH for _, bt in result)


def test_each_paragraph_its_own_chunk():
    tts = _ConcreteTTS()
    text = "\n".join(["a" * 60, "b" * 60, "c" * 60])
    result = tts._chunk_text_with_breaks(text, max_chars=80)
    assert result == [("a" * 60, BREAK_PARAGRAPH), ("b" * 60, BREAK_PARAGRAPH), ("c" * 60, BREAK_PARAGRAPH)]


def test_sentence_split_inside_long_paragraph():
    tts = _ConcreteTTS()
    sentences = [("word " * 10).strip() + "." for _ in range(6)]  # 50 chars each
    long_para = " ".join(sentences)
    outro = "outro " * 4 + "end"  # too long to share a chunk with the last sentences
    text = "short intro\n" + long_para + "\n" + outro
    result = tts._chunk_text_with_breaks(text, max_chars=120)

    chunks = [c for c, _ in result]
    breaks = [b for _, b in result]
    assert all(len(c) <= 120 for c in chunks)
    assert chunks[0] == "short intro"
    assert chunks[-1] == outro
    # intro ends at paragraph boundary, the long paragraph is cut at sentence ends,
    # and its final piece ends at the paragraph boundary before the outro.
    assert breaks[0] == BREAK_PARAGRAPH
    assert breaks[1:-2] and all(b == BREAK_SENTENCE for b in breaks[1:-2])
    assert breaks[-2] == BREAK_PARAGRAPH


def test_word_split_of_overlong_sentence_is_sentence_break():
    tts = _ConcreteTTS()
    text = " ".join(["word"] * 100)  # one 499-char "sentence", no punctuation
    result = tts._chunk_text_with_breaks(text, max_chars=100)
    assert len(result) > 1
    assert all(bt == BREAK_SENTENCE for _, bt in result[:-1])
    assert result[-1][1] == BREAK_PARAGRAPH  # last chunk: irrelevant, reported as paragraph
    # no words lost or reordered
    assert " ".join(c for c, _ in result).split() == text.split()


def test_no_empty_chunks_when_first_word_or_sentence_exceeds_limit():
    tts = _ConcreteTTS()
    # First "word" longer than max_chars used to emit an empty chunk.
    text = "x" * 50 + " short words here and more words to fill the paragraph up"
    result = tts._chunk_text_with_breaks(text, max_chars=20)
    assert all(c.strip() for c, _ in result)

    # First sentence longer than max_chars (has sentence split path too).
    text2 = "y" * 60 + ". " + "z" * 10 + "."
    result2 = tts._chunk_text_with_breaks(text2, max_chars=30)
    assert all(c.strip() for c, _ in result2)
    assert "".join(c for c, _ in result2).replace(" ", "").replace("\n", "").startswith("y" * 60)


def test_blank_lines_ignored_and_chunk_text_matches_with_breaks():
    tts = _ConcreteTTS()
    text = "\n\n".join(["p" * 50] * 5) + "\n   \n" + ("q " * 40)
    assert tts._chunk_text(text, 120) == [c for c, _ in tts._chunk_text_with_breaks(text, 120)]
    assert all(c.strip() for c in tts._chunk_text(text, 120))


# ---------------------------------------------------------------------------
# Gap planning
# ---------------------------------------------------------------------------

def test_plan_gaps_by_break_type():
    breaks = [BREAK_PARAGRAPH, BREAK_SENTENCE, BREAK_PARAGRAPH, BREAK_SENTENCE]
    # last chunk's break type is ignored -> n-1 gaps
    assert plan_gaps(breaks, 700, 300) == [700, 300, 700]


def test_plan_gaps_single_chunk_and_empty():
    assert plan_gaps([BREAK_PARAGRAPH], 700, 300) == []
    assert plan_gaps([], 700, 300) == []


def test_plan_gaps_zero_and_unknown():
    assert plan_gaps([BREAK_PARAGRAPH, BREAK_SENTENCE, BREAK_PARAGRAPH], 0, 0) == [0, 0]
    with pytest.raises(ValueError):
        plan_gaps(["bogus", BREAK_PARAGRAPH], 1, 1)


def test_engine_plan_gaps_uses_engine_settings():
    tts = _ConcreteTTS()
    tts._set_chunk_gaps(500, 100)
    assert tts._plan_gaps([BREAK_SENTENCE, BREAK_PARAGRAPH, BREAK_PARAGRAPH]) == [100, 500]


def test_validate_gap_ms():
    assert validate_gap_ms(0) == 0
    assert validate_gap_ms(250.0) == 250
    for bad in (-1, "300", None, True, float("nan")):
        with pytest.raises(ValueError):
            validate_gap_ms(bad)


def test_engine_defaults_and_explicit_gaps():
    assert ElevenLabsTTS(api_key="k").chunk_gap_paragraph_ms == 700
    assert ElevenLabsTTS(api_key="k").chunk_gap_sentence_ms == 300
    o = OpenAITTS(api_key="k", chunk_gap_paragraph_ms=0, chunk_gap_sentence_ms=50)
    assert (o.chunk_gap_paragraph_ms, o.chunk_gap_sentence_ms) == (0, 50)
    g = GoogleCloudTTS(chunk_gap_paragraph_ms=10, chunk_gap_sentence_ms=20)
    assert (g.chunk_gap_paragraph_ms, g.chunk_gap_sentence_ms) == (10, 20)
    with patch("google.auth.default", side_effect=Exception("no creds")):
        gem = GeminiTTS("m", "v", chunk_gap_paragraph_ms=11, chunk_gap_sentence_ms=22)
    assert (gem.chunk_gap_paragraph_ms, gem.chunk_gap_sentence_ms) == (11, 22)
    with pytest.raises(ValueError):
        OpenAITTS(api_key="k", chunk_gap_paragraph_ms=-5)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(yaml_content: str) -> PipelineConfig:
    with patch("builtins.open", mock_open(read_data=yaml_content)), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        return PipelineConfig("config.yaml")


def test_config_chunk_gap_defaults_when_missing():
    config = _load_config('halachic_section: "Yoreh De\'ah"\ngenerator:\n  engine: "gemini"\n')
    assert config.tts_chunk_gap_paragraph_ms == 700
    assert config.tts_chunk_gap_sentence_ms == 300


def test_config_chunk_gap_explicit_values():
    config = _load_config(
        'halachic_section: "Yoreh De\'ah"\n'
        'tts:\n  engine: "openai"\n  chunk_gap_paragraph_ms: 1000\n  chunk_gap_sentence_ms: 0\n'
    )
    assert config.tts_chunk_gap_paragraph_ms == 1000
    assert config.tts_chunk_gap_sentence_ms == 0


def test_config_chunk_gap_invalid_raises():
    with pytest.raises(ValueError):
        _load_config(
            'halachic_section: "Yoreh De\'ah"\ntts:\n  chunk_gap_paragraph_ms: -10\n'
        )


def test_repo_config_yaml_has_gap_settings():
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        tts = yaml.safe_load(f)["tts"]
    assert tts["chunk_gap_paragraph_ms"] == 700
    assert tts["chunk_gap_sentence_ms"] == 300


@patch("pipeline.tts.OpenAI")
def test_factory_passes_gaps_to_engine(_mock_openai):
    config = MagicMock(spec=PipelineConfig)
    config.tts_engine = "openai"
    config.openai_api_key = "k"
    config.openai_tts_settings = {"voice": "alloy", "model": "tts-1"}
    config.ssl_verify = True
    config.tts_chunk_gap_paragraph_ms = 900
    config.tts_chunk_gap_sentence_ms = 120
    engine = create_tts_engine(config)
    assert (engine.chunk_gap_paragraph_ms, engine.chunk_gap_sentence_ms) == (900, 120)


def test_factory_passes_gaps_to_gemini():
    config = MagicMock(spec=PipelineConfig)
    config.tts_engine = "gemini"
    config.gemini_tts_model = "m"
    config.gemini_tts_voice = "Achird"
    config.gemini_api_key = "k"
    config.tts_chunk_gap_paragraph_ms = 640
    config.tts_chunk_gap_sentence_ms = 0
    with patch("google.auth.default", side_effect=Exception("no creds")):
        engine = create_tts_engine(config)
    assert isinstance(engine, GeminiTTS)
    assert (engine.chunk_gap_paragraph_ms, engine.chunk_gap_sentence_ms) == (640, 0)


# ---------------------------------------------------------------------------
# Gemini PCM silence
# ---------------------------------------------------------------------------

def test_gemini_silence_pcm_length_and_even():
    assert len(GeminiTTS._silence_pcm(1000)) == 48000  # 24000 Hz * 2 bytes
    assert len(GeminiTTS._silence_pcm(700)) == 33600
    assert len(GeminiTTS._silence_pcm(0)) == 0
    for ms in (1, 3, 7, 333):
        n = len(GeminiTTS._silence_pcm(ms))
        assert n % 2 == 0
    assert set(GeminiTTS._silence_pcm(10)) <= {0}


def test_gemini_assemble_pcm_inserts_gaps():
    with patch("google.auth.default", side_effect=Exception("no creds")):
        gem = GeminiTTS("m", "v")
    out = gem._assemble_pcm([b"\x01\x01", b"\x02\x02", b"\x03\x03"], [1000, 0])
    assert out == b"\x01\x01" + bytes(48000) + b"\x02\x02" + b"\x03\x03"
    assert gem._assemble_pcm([b"\x01\x01"], []) == b"\x01\x01"


# ---------------------------------------------------------------------------
# Filter graph + ffmpeg integration
# ---------------------------------------------------------------------------

def test_build_gap_filter_graph_structure():
    graph = BaseTTS._build_gap_filter_graph([700, 0, 300])
    assert "[0:a]" in graph and "[3:a]" in graph
    assert "[s0]" in graph and "[s2]" in graph
    assert "[s1]" not in graph  # zero gap -> no silence segment
    assert "atrim=duration=0.700" in graph and "atrim=duration=0.300" in graph
    assert "concat=n=6:v=0:a=1[out]" in graph  # 4 chunks + 2 silences
    assert "aresample=44100" in graph


imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")


def _ffmpeg():
    return imageio_ffmpeg.get_ffmpeg_exe()


def _make_tone_mp3(path, seconds: float, sample_rate: int = 44100) -> None:
    subprocess.run(
        [_ffmpeg(), "-y", "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={sample_rate}",
         "-t", str(seconds), "-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", str(path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _duration_seconds(path) -> float:
    """Decodes to raw PCM and measures length (no ffprobe needed)."""
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "44100", "-"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    return len(proc.stdout) / 2 / 44100


def test_merge_with_gaps_duration(tmp_path):
    a, b, c = tmp_path / "a.mp3", tmp_path / "b.mp3", tmp_path / "c.mp3"
    _make_tone_mp3(a, 1.0)
    _make_tone_mp3(b, 1.5)
    _make_tone_mp3(c, 0.5)
    out = tmp_path / "out" / "merged.mp3"
    _ConcreteTTS()._merge_audio_chunks([str(a), str(b), str(c)], str(out), gaps_ms=[700, 300])
    assert out.exists()
    expected = 1.0 + 0.7 + 1.5 + 0.3 + 0.5
    assert _duration_seconds(out) == pytest.approx(expected, abs=0.15)


def test_merge_with_gaps_zero_gap_entry(tmp_path):
    a, b, c = tmp_path / "a.mp3", tmp_path / "b.mp3", tmp_path / "c.mp3"
    for p in (a, b, c):
        _make_tone_mp3(p, 1.0)
    out = tmp_path / "merged.mp3"
    _ConcreteTTS()._merge_audio_chunks([str(a), str(b), str(c)], str(out), gaps_ms=[0, 500])
    assert _duration_seconds(out) == pytest.approx(3.5, abs=0.15)


def test_merge_with_gaps_handles_mixed_sample_rates(tmp_path):
    """Google/OpenAI return 24 kHz MP3s; ElevenLabs returns 44.1 kHz. They must mix."""
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    _make_tone_mp3(a, 1.0, sample_rate=24000)
    _make_tone_mp3(b, 1.0, sample_rate=44100)
    out = tmp_path / "merged.mp3"
    _ConcreteTTS()._merge_audio_chunks([str(a), str(b)], str(out), gaps_ms=[700])
    assert _duration_seconds(out) == pytest.approx(2.7, abs=0.15)


def test_merge_without_gaps_uses_stream_copy(tmp_path):
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    _make_tone_mp3(a, 1.0)
    _make_tone_mp3(b, 1.0)
    tts = _ConcreteTTS()
    for gaps in (None, [0]):
        out = tmp_path / f"merged_{gaps}.mp3"
        with patch.object(tts, "_merge_with_gaps") as with_gaps, \
             patch.object(tts, "_merge_stream_copy", wraps=tts._merge_stream_copy) as copy:
            tts._merge_audio_chunks([str(a), str(b)], str(out), gaps_ms=gaps)
            copy.assert_called_once()
            with_gaps.assert_not_called()
        assert _duration_seconds(out) == pytest.approx(2.0, abs=0.15)


def test_merge_rejects_wrong_gap_count(tmp_path):
    with pytest.raises(ValueError):
        _ConcreteTTS()._merge_audio_chunks(["a", "b", "c"], str(tmp_path / "o.mp3"), gaps_ms=[100])


def test_synthesize_passes_planned_gaps_to_merge(tmp_path):
    class _Fake(BaseTTS):
        _chunk_size = 100

        def _synthesize_chunk(self, text):
            return b"x"

    tts = _Fake()
    tts._set_chunk_gaps(700, 300)
    text = "\n".join(["a" * 60, "b" * 60, "c" * 60])
    with patch.object(tts, "_merge_audio_chunks") as merge:
        tts.synthesize(text, str(tmp_path / "o.mp3"))
    assert merge.call_args.kwargs["gaps_ms"] == [700, 700]
