import os
from unittest.mock import MagicMock, patch, mock_open

import pytest

from pipeline.pronunciation import (
    strip_phoneme_tags, extract_phoneme_pairs, find_malformed_tags, validate_tagged_output,
    tag_pronunciation, format_pairs, run_pronunciation_stage, split_phoneme_segments,
    PronunciationTaggingError,
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


# --- validation (failures raise) ---------------------------------------------------

def test_valid_tagging_accepted():
    text, pairs = tag_pronunciation(ORIG, lambda t: TAGGED)
    assert text == TAGGED and len(pairs) == 3


def test_trivial_whitespace_differences_tolerated():
    llm = TAGGED.replace("\n", "  \r\n").rstrip() + "\n\n"
    assert validate_tagged_output(ORIG, llm) is None


def test_changed_text_raises():
    changed = TAGGED.replace("עוסק", "דן")
    with pytest.raises(PronunciationTaggingError, match="text changed outside tags"):
        tag_pronunciation(ORIG, lambda t: changed)


def test_inner_word_changed_raises():
    bad = TAGGED.replace(">טית<", ">טֵית<")
    with pytest.raises(PronunciationTaggingError, match="text changed outside tags"):
        tag_pronunciation(ORIG, lambda t: bad)


def test_llm_exception_raises():
    def boom(_):
        raise RuntimeError("network")
    with pytest.raises(PronunciationTaggingError, match="LLM call failed: network"):
        tag_pronunciation(ORIG, boom)


def test_empty_output_raises():
    with pytest.raises(PronunciationTaggingError, match="empty output"):
        tag_pronunciation(ORIG, lambda t: "  ")


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
    with pytest.raises(PronunciationTaggingError, match="malformed phoneme tag"):
        tag_pronunciation(ORIG, lambda t: llm)


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
  elevenlabs:
    chunk_size: 8000
    context_chars: 500
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
  always_tag: []
  never_tag: []
""")
    assert cfg.pronunciation_enabled is True
    assert cfg.pronunciation_engine == "gemini"
    assert cfg.pronunciation_model_name == "m"
    assert cfg.pronunciation_temperature == 0.2
    assert cfg.pronunciation_instruction == "prompt"


def test_config_missing_section_errors():
    cfg = _config()
    for prop in ("pronunciation_enabled", "pronunciation_engine", "pronunciation_model_name",
                 "pronunciation_temperature", "pronunciation_instruction",
                 "pronunciation_always_tag", "pronunciation_never_tag"):
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


PLACEHOLDER_PROMPT = 'תמיד: {always_tag} | לעולם: {never_tag} | <phoneme alphabet="ipa" ph="{x}">{y}</phoneme>'


def _placeholder_config(always, never):
    return _config(f"""
pronunciation_instruction: '{PLACEHOLDER_PROMPT}'
pronunciation:
  enabled: true
  engine: "gemini"
  model_name: "m"
  temperature: 0.2
  always_tag: {always}
  never_tag: {never}
""")


def test_instruction_placeholders_filled():
    cfg = _placeholder_config('["עובר בהמה", "כותח"]', '["דאורייתא", "דרבנן"]')
    assert cfg.pronunciation_always_tag == ["עובר בהמה", "כותח"]
    assert cfg.pronunciation_never_tag == ["דאורייתא", "דרבנן"]
    assert cfg.pronunciation_instruction == (
        'תמיד: עובר בהמה, כותח | לעולם: דאורייתא, דרבנן | '
        '<phoneme alphabet="ipa" ph="{x}">{y}</phoneme>'  # other braces left untouched
    )


def test_instruction_empty_lists_render_none_word():
    cfg = _placeholder_config("[]", "[]")
    assert cfg.pronunciation_instruction.startswith("תמיד: אין | לעולם: אין | ")


def test_instruction_missing_lists_error():
    cfg = _config("""
pronunciation_instruction: "x {always_tag}"
pronunciation:
  enabled: true
""")
    with pytest.raises(ValueError, match="always_tag"):
        cfg.pronunciation_instruction


@pytest.mark.parametrize("bad", ['"כותח"', '[1, 2]', '[""]'])
def test_instruction_invalid_list_errors(bad):
    cfg = _placeholder_config(bad, "[]")
    with pytest.raises(ValueError, match="always_tag"):
        cfg.pronunciation_always_tag


def test_real_config_yaml_has_pronunciation():
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        cfg = PipelineConfig("config.yaml")
    assert isinstance(cfg.pronunciation_enabled, bool)
    assert "phoneme" in cfg.pronunciation_instruction
    assert "{always_tag}" not in cfg.pronunciation_instruction
    assert "{never_tag}" not in cfg.pronunciation_instruction
    assert "תמיד לעטוף: עובר בהמה, כותח" in cfg.pronunciation_instruction
    assert "לעולם לא לעטוף: דאורייתא, דרבנן" in cfg.pronunciation_instruction


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


def test_main_finalize_tagging_failure_writes_nothing(tmp_path):
    import main as main_mod
    cfg = MagicMock(output_dir=str(tmp_path), section_slug="YD")
    err = PronunciationTaggingError("pronunciation tagging output rejected: bad")
    with patch.object(main_mod, "run_pronunciation_stage", side_effect=err):
        with pytest.raises(main_mod.SimanStageError) as exc:
            main_mod.finalize_and_save_transcript(94, ORIG, cfg, "m")
    assert exc.value.stage == main_mod.STAGE_PRONUNCIATION
    assert "rejected: bad" in exc.value.reason
    assert os.listdir(tmp_path) == []


def test_main_tagging_failure_skips_tts(tmp_path):
    import main as main_mod
    cfg = MagicMock(output_dir=str(tmp_path), section_slug="YD")
    args = MagicMock(skip_tts=False)
    with patch.object(main_mod, "run_pronunciation_stage", side_effect=PronunciationTaggingError("x")), \
         patch.object(main_mod, "create_tts_engine") as tts:
        with pytest.raises(main_mod.SimanStageError):
            main_mod.finalize_and_synthesize(94, ORIG, cfg, args, "m")
    tts.assert_not_called()


def test_main_tts_failure_is_stage_error(tmp_path):
    import main as main_mod
    cfg = MagicMock(pronunciation_enabled=False, output_dir=str(tmp_path), section_slug="YD")
    args = MagicMock(skip_tts=False)
    with patch.object(main_mod, "create_tts_engine", side_effect=RuntimeError("quota")):
        with pytest.raises(main_mod.SimanStageError) as exc:
            main_mod.finalize_and_synthesize(94, ORIG, cfg, args, "m")
    assert exc.value.stage == main_mod.STAGE_TTS and exc.value.reason == "quota"


def test_run_summary_format():
    import main as main_mod
    summary = main_mod.RunSummary()
    summary.success(94)
    summary.failure(95, main_mod.STAGE_PRONUNCIATION, "boom")
    summary.success(96)
    assert summary.has_failures
    assert summary.format() == (
        "===== Run summary =====\n"
        "Succeeded (2/3): 94, 96\n"
        "Failed (1/3): 95\n"
        "  - Siman 95 | stage: Stage 3.4 (Pronunciation tagging) | reason: boom"
    )
    empty = main_mod.RunSummary()
    empty.success(1)
    assert not empty.has_failures
    assert "Failed (0/1): none" in empty.format()


def _run_main_sync(tmp_path, monkeypatch, polish):
    """Runs main.main() in sync mode for simanim 94,95 with cached drafts and relations."""
    import main as main_mod
    cfg = MagicMock(section_slug="YD", drafts_dir=str(tmp_path), relations_dir=str(tmp_path),
                    output_dir=str(tmp_path), pronunciation_enabled=False)
    for s in (94, 95):
        (tmp_path / f"YD_Siman_{s}_m_draft.txt").write_text(f"draft {s}", encoding="utf-8")
    (tmp_path / "YD_Relations_94_95_m.txt").write_text("rel", encoding="utf-8")
    gen = MagicMock(model_name="m")
    gen.polish_siman_script.side_effect = polish
    monkeypatch.setattr("sys.argv", ["main.py", "94,95", "--skip-tts"])
    with patch.object(main_mod, "PipelineConfig", return_value=cfg), \
         patch.object(main_mod, "create_generator_engine", return_value=gen), \
         patch.object(main_mod, "SefariaExtractor"), \
         patch("pipeline.evaluator.evaluate_draft", return_value=MagicMock()):
        main_mod.main()


def test_main_continues_after_siman_failure_and_exits_nonzero(tmp_path, monkeypatch, caplog):
    def polish(draft, rel, instr):
        if draft == "draft 94":
            raise RuntimeError("LLM down")
        return "polished"
    with pytest.raises(SystemExit) as exc:
        _run_main_sync(tmp_path, monkeypatch, polish)
    assert exc.value.code == 1
    assert (tmp_path / "YD_Siman_95_m_transcript.txt").exists()
    assert not (tmp_path / "YD_Siman_94_m_transcript.txt").exists()
    assert "Succeeded (1/2): 95" in caplog.text
    assert "Siman 94 | stage: Stage 3 (Polishing) | reason: LLM down" in caplog.text


def test_main_all_succeed_exits_zero(tmp_path, monkeypatch, caplog):
    _run_main_sync(tmp_path, monkeypatch, lambda d, r, i: "polished")
    assert "Succeeded (2/2): 94, 95" in caplog.text


def test_tag_script_errors_without_writing(tmp_path, monkeypatch, capsys):
    import scripts.tag_pronunciation as script
    monkeypatch.chdir(tmp_path)  # script chdirs to the repo root; restore cwd afterwards
    src = tmp_path / "x_transcript.txt"
    src.write_text(ORIG, encoding="utf-8")
    gen = MagicMock(model_name="m")
    gen.generate_text.return_value = TAGGED.replace("עוסק", "דן")
    monkeypatch.setattr("pipeline.config.PipelineConfig", MagicMock())
    monkeypatch.setattr("pipeline.factory.create_pronunciation_generator", lambda c: gen)
    with pytest.raises(SystemExit) as exc:
        script.main([str(src)])
    assert exc.value.code == 1
    assert "text changed outside tags" in capsys.readouterr().err
    assert sorted(os.listdir(tmp_path)) == ["x_transcript.txt"]


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
