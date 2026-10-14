import os
from unittest.mock import MagicMock, patch, mock_open

import pytest

from pipeline.pronunciation import (
    strip_phoneme_tags, extract_phoneme_pairs, find_malformed_tags, validate_tagged_output,
    tag_pronunciation, format_pairs, run_pronunciation_stage, split_phoneme_segments,
)
from pipeline.apply_nikkud import apply_nikkud_to_abbreviations
from pipeline.tts import BaseTTS, _split_outside_phoneme

TET = '<phoneme alphabet="ipa" ph="tet">טית</phoneme>'
PAT = '<phoneme alphabet="ipa" ph="pat">פת</phoneme>'
SOFER = '<phoneme alphabet="ipa" ph="taˈut soˈfer">טעות סופר</phoneme>'

ORIG = "סימן טית עוסק בדין פת.\nזו טעות סופר בדברי הש\"ך.\n"
TAGGED = f"סימן {TET} עוסק בדין {PAT}.\nזו {SOFER} בדברי הש\"ך.\n"


# --- strip / pairs -----------------------------------------------------------

def test_strip_keeps_inner_word():
    assert strip_phoneme_tags(TAGGED) == ORIG


def test_extract_pairs():
    assert extract_phoneme_pairs(TAGGED) == [("טית", "tet"), ("פת", "pat"), ("טעות סופר", "taˈut soˈfer")]
    assert format_pairs(extract_phoneme_pairs(TAGGED)).splitlines()[0] == "טית\ttet"


# --- validation / fallback ---------------------------------------------------

def test_valid_tagging_accepted():
    text, pairs = tag_pronunciation(ORIG, lambda t: TAGGED)
    assert text == TAGGED and len(pairs) == 3


def test_trivial_whitespace_differences_tolerated():
    llm = TAGGED.replace("\n", "  \r\n").rstrip() + "\n\n"
    assert validate_tagged_output(ORIG, llm) is None


def test_changed_text_falls_back_to_original(caplog):
    changed = TAGGED.replace("עוסק", "דן")
    text, pairs = tag_pronunciation(ORIG, lambda t: changed)
    assert text == ORIG and pairs == []
    assert "text changed outside tags" in caplog.text


def test_inner_word_changed_falls_back():
    bad = TAGGED.replace(">טית<", ">טֵית<")
    assert tag_pronunciation(ORIG, lambda t: bad) == (ORIG, [])


def test_llm_exception_falls_back():
    def boom(_):
        raise RuntimeError("network")
    assert tag_pronunciation(ORIG, boom) == (ORIG, [])


def test_empty_output_falls_back():
    assert tag_pronunciation(ORIG, lambda t: "  ") == (ORIG, [])


@pytest.mark.parametrize("bad_tag", [
    '<phoneme alphabet="ipa" ph="tet">טית',                       # unclosed
    'טית</phoneme>',                                              # stray close
    '<phoneme alphabet="ipa">טית</phoneme>',                      # missing ph
    '<phoneme alphabet="ipa" ph="">טית</phoneme>',                # empty ph
    '<phoneme alphabet="ipa" ph="x"><phoneme alphabet="ipa" ph="tet">טית</phoneme></phoneme>',  # nested
    '<phoneme alphabet="x-sampa" ph="tet">טית</phoneme>',          # wrong alphabet
])
def test_malformed_tags_rejected(bad_tag):
    llm = f"סימן {bad_tag} עוסק בדין פת.\nזו טעות סופר בדברי הש\"ך.\n"
    assert find_malformed_tags(llm)
    assert tag_pronunciation(ORIG, lambda t: llm) == (ORIG, [])


# --- nikkud skips tag spans --------------------------------------------------

def test_nikkud_skips_tag_spans():
    tag = '<phoneme alphabet="ipa" ph="ʃaχ">הש"ך</phoneme>'
    text = f'סימן צדי {tag} והש"ך אומר'
    out = apply_nikkud_to_abbreviations(text)
    assert tag in out                       # inner word and ph untouched
    assert out.endswith('והשַׁךְ אומר')       # outside text still nikkud'd
    assert out.startswith("סימן צָדִי ")


def test_nikkud_unchanged_without_tags():
    assert apply_nikkud_to_abbreviations('הש"ך') == "השַׁךְ"


def test_split_segments_roundtrip():
    segs = split_phoneme_segments(TAGGED)
    assert "".join(s for s, _ in segs) == TAGGED
    assert [s for s, is_tag in segs if is_tag] == [TET, PAT, SOFER]


# --- TTS chunker never splits inside a tag ---------------------------------

class _DummyTTS(BaseTTS):
    _chunk_size = 50

    def _synthesize_chunk(self, text):
        return b""


def _assert_tags_intact(chunks):
    for chunk in chunks:
        assert chunk.count("<phoneme") == chunk.count("</phoneme>")


def test_split_outside_phoneme():
    assert _split_outside_phoneme(f"a {SOFER} b", " ") == ["a", SOFER, "b"]


def test_chunker_word_split_keeps_tags():
    text = " ".join(["מילה", SOFER] * 20)
    chunks = _DummyTTS()._chunk_text(text, max_chars=60)
    assert len(chunks) > 1
    _assert_tags_intact(chunks)
    assert "".join(strip_phoneme_tags(c).replace(" ", "") for c in chunks) == strip_phoneme_tags(text).replace(" ", "")


def test_chunker_sentence_split_keeps_tags():
    tag = '<phoneme alphabet="ipa" ph="a. b">א ב</phoneme>'  # '.' + space inside the tag
    text = " ".join([f"משפט {tag} סוף."] * 15)
    _assert_tags_intact(_DummyTTS()._chunk_text(text, max_chars=80))


# --- config properties -------------------------------------------------------

BASE_YAML = """
halachic_section: "Yoreh De'ah"
generator:
  engine: "gemini"
tts:
  chunk_gap_paragraph_ms: 700
  chunk_gap_sentence_ms: 300
"""


def _config(extra=""):
    from pipeline.config import PipelineConfig
    with patch("builtins.open", mock_open(read_data=BASE_YAML + extra)), \
         patch("os.path.exists", return_value=True), patch("os.makedirs"):
        return PipelineConfig("config.yaml")


def test_config_pronunciation_properties():
    cfg = _config("""
pronunciation_instruction: "prompt"
pronunciation:
  enabled: true
  engine: "Gemini"
  model_name: "m"
  temperature: 0.2
""")
    assert cfg.pronunciation_enabled is True
    assert cfg.pronunciation_engine == "gemini"
    assert cfg.pronunciation_model_name == "m"
    assert cfg.pronunciation_temperature == 0.2
    assert cfg.pronunciation_instruction == "prompt"


def test_config_missing_section_errors():
    cfg = _config()
    for prop in ("pronunciation_enabled", "pronunciation_engine", "pronunciation_model_name",
                 "pronunciation_temperature", "pronunciation_instruction"):
        with pytest.raises(ValueError, match="pronunciation"):
            getattr(cfg, prop)


def test_config_missing_key_errors():
    cfg = _config("""
pronunciation:
  enabled: true
  engine: "gemini"
""")
    with pytest.raises(ValueError, match="model_name"):
        cfg.pronunciation_model_name
    with pytest.raises(ValueError, match="temperature"):
        cfg.pronunciation_temperature


def test_config_invalid_values():
    cfg = _config("""
pronunciation:
  enabled: "yes"
  engine: "claude"
""")
    with pytest.raises(ValueError):
        cfg.pronunciation_enabled
    with pytest.raises(ValueError):
        cfg.pronunciation_engine


def test_real_config_yaml_has_pronunciation():
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        cfg = PipelineConfig("config.yaml")
    assert isinstance(cfg.pronunciation_enabled, bool)
    assert "phoneme" in cfg.pronunciation_instruction


# --- stage wiring ------------------------------------------------------------

def test_stage_skipped_when_disabled():
    cfg = MagicMock(pronunciation_enabled=False)
    factory = MagicMock()
    assert run_pronunciation_stage(ORIG, cfg, generator_factory=factory) is None
    factory.assert_not_called()


def test_stage_runs_when_enabled():
    cfg = MagicMock(pronunciation_enabled=True, pronunciation_instruction="INSTR", pronunciation_engine="gemini")
    gen = MagicMock(model_name="m")
    gen.generate_text.return_value = TAGGED
    text, pairs = run_pronunciation_stage(ORIG, cfg, generator_factory=lambda c: gen)
    assert text == TAGGED and len(pairs) == 3
    gen.generate_text.assert_called_once_with("INSTR", ORIG)


def test_factory_creates_pronunciation_generator():
    from pipeline.factory import create_pronunciation_generator
    cfg = MagicMock(pronunciation_engine="gemini", pronunciation_model_name="pm",
                    pronunciation_temperature=0.2, gemini_api_key="k", ssl_verify=True)
    with patch("pipeline.factory.GeminiScriptGenerator") as gem:
        create_pronunciation_generator(cfg)
    gem.assert_called_once_with(api_key="k", model_name="pm", temperature=0.2, ssl_verify=True)


def test_main_finalize_disabled_skips_pronunciation_files(tmp_path):
    import main as main_mod
    cfg = MagicMock(pronunciation_enabled=False, output_dir=str(tmp_path), section_slug="YD")
    out = main_mod.finalize_and_save_transcript(94, 'הש"ך', cfg, "m")
    assert out == "השַׁךְ"
    names = os.listdir(tmp_path)
    assert "YD_Siman_94_m_transcript.txt" in names
    assert not any("pronunciation" in n for n in names)


def test_main_finalize_enabled_saves_outputs(tmp_path):
    import main as main_mod
    cfg = MagicMock(output_dir=str(tmp_path), section_slug="YD")
    with patch.object(main_mod, "run_pronunciation_stage", return_value=(TAGGED, [("טית", "tet")])):
        out = main_mod.finalize_and_save_transcript(94, ORIG, cfg, "m")
    assert TET in out
    assert (tmp_path / "YD_Siman_94_m_pronunciation_tags.txt").read_text(encoding="utf-8") == "טית\ttet\n"
    assert (tmp_path / "YD_Siman_94_m_pronunciation.txt").read_text(encoding="utf-8") == TAGGED


# --- scripts ----------------------------------------------------------------

def test_tag_script_args(tmp_path):
    from scripts.tag_pronunciation import parse_args, tags_path_for
    args = parse_args([str(tmp_path / "x_transcript.txt"), "--apply-nikkud"])
    assert args.output == str(tmp_path / "x_transcript_tagged.txt")
    assert args.apply_nikkud is True
    assert tags_path_for(args.output) == str(tmp_path / "x_transcript_tagged_tags.txt")


def test_tts_script_args(tmp_path):
    from scripts.tts_from_file import parse_args
    args = parse_args([str(tmp_path / "a.txt"), "--output", str(tmp_path / "a.mp3")])
    assert args.output == str(tmp_path / "a.mp3")
