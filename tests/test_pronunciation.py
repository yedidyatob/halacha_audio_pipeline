import os
from unittest.mock import MagicMock, patch, mock_open

import pytest

from pipeline.pronunciation import (
    strip_nikkud, has_nikkud, find_markup, validate_vocalized_output, changed_words,
    vocalize_pronunciation, format_changes, parse_vocalized_response,
    PronunciationVocalizer, create_pronunciation_vocalizer, PronunciationError,
)
from pipeline.apply_nikkud import apply_nikkud_to_abbreviations

# Input after the dictionary nikkud stage: השַׁךְ is already vocalized.
ORIG = "סימן טית עוסק בדין פת.\nזו טעות סופר בדברי השַׁךְ.\n"
VOCALIZED = "סימן טֵית עוסק בדין פַּת.\nזו טעות סוֹפֵר בדברי השַׁךְ.\n"
CHANGES = [("טית", "טֵית"), ("פת", "פַּת"), ("סופר", "סוֹפֵר")]


# --- nikkud helpers -----------------------------------------------------------

def test_strip_nikkud_keeps_letters_and_punctuation():
    assert strip_nikkud(VOCALIZED) == strip_nikkud(ORIG) == "סימן טית עוסק בדין פת.\nזו טעות סופר בדברי השך.\n"
    assert strip_nikkud("בית־דין׀") == "בית־דין׀"  # maqaf / paseq are not nikkud
    assert has_nikkud("טֵית") and not has_nikkud("טית")


def test_changed_words_and_format():
    assert changed_words(ORIG, VOCALIZED) == CHANGES  # trailing punctuation trimmed
    assert format_changes(CHANGES).splitlines()[0] == "טית\tטֵית"


# --- validation (failures raise) -------------------------------------------------

def test_valid_output_accepted():
    text, changes = vocalize_pronunciation(ORIG, lambda t: VOCALIZED)
    assert text == VOCALIZED and changes == CHANGES


def test_no_changes_is_valid():
    assert vocalize_pronunciation(ORIG, lambda t: ORIG) == (ORIG, [])


def test_trivial_whitespace_differences_tolerated():
    llm = VOCALIZED.replace("\n", "  \r\n").rstrip() + "\n\n"
    assert validate_vocalized_output(ORIG, llm) is None


@pytest.mark.parametrize("bad", [
    VOCALIZED.replace("סימן ", "סימן  "),      # inner whitespace changed
    VOCALIZED.replace("זו ", ""),              # word dropped
    VOCALIZED.replace("פַּת.\n", "פַּת. "),     # line joined
])
def test_layout_change_raises(bad):
    with pytest.raises(PronunciationError, match="text changed"):
        vocalize_pronunciation(ORIG, lambda t: bad)


# --- spelling: ו / י may be added or removed, except as a word's first letter ------

MATRES_ORIG = "אסור לאכול עובר בהמה, ויש שהקילו.\n"


@pytest.mark.parametrize("word, respelled", [
    ("עובר", "עֻבָּר"),        # ו removed mid-word
    ("שהקילו", "שֶׁהֵקֵלּוּ"),  # י removed mid-word
    ("לאכול", "לֶאֱכוֹל"),     # unchanged letters
    ("בהמה,", "בְּהֵימָה,"),    # י added mid-word (punctuation kept)
    ("שהקילו", "שהקלו"),       # respelled without nikkud
])
def test_matres_change_mid_word_accepted(word, respelled):
    out = MATRES_ORIG.replace(word, respelled)
    text, changes = vocalize_pronunciation(MATRES_ORIG, lambda t: out)
    assert text == out
    assert changes == [(word.strip(","), respelled.strip(","))]


def test_respelled_words_listed_in_changes():
    out = "אסור לאכול עֻבָּר בהמה, ויש שֶׁהֵקֵלּוּ.\n"
    _, changes = vocalize_pronunciation(MATRES_ORIG, lambda t: out)
    assert changes == [("עובר", "עֻבָּר"), ("שהקילו", "שֶׁהֵקֵלּוּ")]
    assert format_changes(changes) == "עובר\tעֻבָּר\nשהקילו\tשֶׁהֵקֵלּוּ\n"


@pytest.mark.parametrize("word, bad", [
    ("ויש", "יֵשׁ"),           # conjunction ו removed
    ("עובר", "וְעֻבָּר"),       # ו added at word start
    ("אסור", "יאסור"),        # י added at word start
])
def test_matres_change_at_word_start_rejected(word, bad):
    with pytest.raises(PronunciationError, match=f"{word} -> {bad} \\(ו/י added or removed at the start"):
        vocalize_pronunciation(MATRES_ORIG, lambda t: MATRES_ORIG.replace(word, bad))


def test_matres_after_maqaf_or_in_abbreviation():
    orig = 'בית־דין אמר ר"י\n'
    assert validate_vocalized_output(orig, 'בית־דין אמר ר"יו\n') is None  # after gershayim: not word start
    assert "at the start" in validate_vocalized_output(orig, 'בית־ודין אמר ר"י\n')  # after maqaf: word start


@pytest.mark.parametrize("word, bad", [
    ("עוסק", "דן"),            # word replaced
    ("טֵית", "טֵאית"),          # other letter added while vocalizing
    ("סוֹפֵר", "סוֹפֵרה"),       # other letter added at the end
    ("סוֹפֵר", "סוֹפֵק"),        # letter substituted
    ("פַּת.", "פַּת,."),         # punctuation added
])
def test_other_letter_change_rejected(word, bad):
    with pytest.raises(PronunciationError, match="letters changed beyond adding/removing ו/י"):
        vocalize_pronunciation(ORIG, lambda t: VOCALIZED.replace(word, bad))


@pytest.mark.parametrize("bad", [
    "הַשַּׁךְ",   # nikkud added to a vocalized word
    "השך",      # nikkud removed from it
    "השַׁיךְ",   # ו/י added to it (allowed only in unvocalized words)
])
def test_already_vocalized_word_must_stay_identical(bad):
    with pytest.raises(PronunciationError, match=f"השַׁךְ\\. -> {bad}\\. \\(already vocalized, must stay identical\\)"):
        vocalize_pronunciation(ORIG, lambda t: VOCALIZED.replace("השַׁךְ", bad))


def test_markup_rejected():
    bad = VOCALIZED.replace("טֵית", '<phoneme alphabet="ipa" ph="tet">טית</phoneme>')
    assert find_markup(bad)
    with pytest.raises(PronunciationError, match="markup in output"):
        vocalize_pronunciation(ORIG, lambda t: bad)


def test_llm_exception_raises():
    def boom(_):
        raise RuntimeError("network")
    with pytest.raises(PronunciationError, match="LLM call failed: network"):
        vocalize_pronunciation(ORIG, boom)


def test_empty_output_raises():
    with pytest.raises(PronunciationError, match="empty output"):
        vocalize_pronunciation(ORIG, lambda t: "  ")


# --- dictionary nikkud stage ---------------------------------------------------

def test_dictionary_respells_letter_name_he_in_letter_context():
    assert apply_nikkud_to_abbreviations("סימן הא") == "סימן הֵי"
    assert apply_nikkud_to_abbreviations("צדי הא") == "צָדִי הֵי"
    assert apply_nikkud_to_abbreviations("הא דתניא") == "הא דתניא"  # not a letter name


def test_dictionary_nikkud_leaves_llm_vocalized_words_alone():
    text = 'סימן טֵית והשַׁךְ אומר הש"ך'
    out = apply_nikkud_to_abbreviations(text)
    assert out == 'סימן טֵית והשַׁךְ אומר השַׁךְ'
    assert apply_nikkud_to_abbreviations(out) == out  # idempotent


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
  always_vocalize: []
  never_vocalize: []
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
                 "pronunciation_always_vocalize", "pronunciation_never_vocalize"):
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


PLACEHOLDER_PROMPT = 'תמיד: {always_vocalize} | לעולם: {never_vocalize} | <phoneme alphabet="ipa" ph="{x}">{y}</phoneme>'


def _placeholder_config(always, never):
    return _config(f"""
pronunciation_instruction: '{PLACEHOLDER_PROMPT}'
pronunciation:
  enabled: true
  engine: "gemini"
  model_name: "m"
  temperature: 0.2
  always_vocalize: {always}
  never_vocalize: {never}
""")


def test_instruction_placeholders_filled():
    cfg = _placeholder_config('["עובר בהמה", "כותח"]', '["דאורייתא", "דרבנן"]')
    assert cfg.pronunciation_always_vocalize == ["עובר בהמה", "כותח"]
    assert cfg.pronunciation_never_vocalize == ["דאורייתא", "דרבנן"]
    assert cfg.pronunciation_instruction == (
        'תמיד: עובר בהמה, כותח | לעולם: דאורייתא, דרבנן | '
        '<phoneme alphabet="ipa" ph="{x}">{y}</phoneme>'  # other braces left untouched
    )


def test_instruction_empty_lists_render_none_word():
    cfg = _placeholder_config("[]", "[]")
    assert cfg.pronunciation_instruction.startswith("תמיד: אין | לעולם: אין | ")


def test_instruction_missing_lists_error():
    cfg = _config("""
pronunciation_instruction: "x {always_vocalize}"
pronunciation:
  enabled: true
""")
    with pytest.raises(ValueError, match="always_vocalize"):
        cfg.pronunciation_instruction


@pytest.mark.parametrize("bad", ['"כותח"', '[1, 2]', '[""]'])
def test_instruction_invalid_list_errors(bad):
    cfg = _placeholder_config(bad, "[]")
    with pytest.raises(ValueError, match="always_vocalize"):
        cfg.pronunciation_always_vocalize


def test_real_config_yaml_has_pronunciation():
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        cfg = PipelineConfig("config.yaml")
    assert isinstance(cfg.pronunciation_enabled, bool)
    instruction = cfg.pronunciation_instruction
    assert "ניקוד" in instruction and "phoneme" not in instruction and "IPA" not in instruction
    assert "{always_vocalize}" not in instruction and "{never_vocalize}" not in instruction
    assert "תמיד לנקד: עובר בהמה, כותח" in instruction
    assert "לעולם לא לנקד: דאורייתא, דרבנן" in instruction


# --- stage wiring ------------------------------------------------------------

def _vocalizer_with_gen(level="low", temperature=0.2):
    cfg = MagicMock(pronunciation_instruction="INSTR", pronunciation_thinking_level=level, pronunciation_engine="gemini")
    gen = MagicMock(model_name="m", temperature=temperature)
    return create_pronunciation_vocalizer(cfg, generator_factory=lambda c: gen), gen


def test_vocalizer_sync_uses_instruction_and_thinking_level():
    vocalizer, gen = _vocalizer_with_gen("low")
    gen.generate_text.return_value = VOCALIZED
    assert vocalizer.vocalize(ORIG) == (VOCALIZED, CHANGES)
    gen.generate_text.assert_called_once_with("INSTR", ORIG, thinking_level="low")


def test_vocalizer_batch_request_uses_own_model_settings():
    vocalizer, gen = _vocalizer_with_gen("medium", temperature=0.35)
    gen.submit_batch.return_value = "job-1"
    assert vocalizer.submit_batch(ORIG, custom_id="c", cache_dir="cache") == "job-1"
    gen.submit_batch.assert_called_once_with(
        system_instruction="INSTR", user_prompt=ORIG, temperature=0.35,
        custom_id="c", cache_dir="cache", thinking_level="medium",
    )


def test_vocalizer_batch_result_validated_like_sync():
    vocalizer, gen = _vocalizer_with_gen()
    gen.get_batch_result.return_value = VOCALIZED
    assert vocalizer.batch_result(ORIG, "job") == (VOCALIZED, CHANGES)
    gen.get_batch_result.return_value = VOCALIZED.replace("עוסק", "דן")
    with pytest.raises(PronunciationError, match="עוסק -> דן"):
        vocalizer.batch_result(ORIG, "job")
    gen.get_batch_result.side_effect = ValueError("Gemini batch returned empty text response.")
    with pytest.raises(PronunciationError, match="batch result unavailable: Gemini batch returned empty"):
        vocalizer.batch_result(ORIG, "job")


def test_parse_vocalized_response_shared_validation():
    assert parse_vocalized_response(ORIG, VOCALIZED.replace("\n", "\r\n"))[0] == VOCALIZED
    with pytest.raises(PronunciationError, match="empty output"):
        parse_vocalized_response(ORIG, "")


def test_vocalizer_supports_batch():
    from pipeline.generator import BatchCapableGenerator
    assert PronunciationVocalizer(MagicMock(spec=BatchCapableGenerator), "i", None).supports_batch
    assert not PronunciationVocalizer(object(), "i", None).supports_batch


def test_factory_creates_pronunciation_generator():
    from pipeline.factory import create_pronunciation_generator
    cfg = MagicMock(pronunciation_engine="gemini", pronunciation_model_name="pm",
                    pronunciation_temperature=0.2, gemini_api_key="k", ssl_verify=True,
                    pronunciation_thinking_level="low")
    with patch("pipeline.factory.GeminiScriptGenerator") as gem:
        create_pronunciation_generator(cfg)
    gem.assert_called_once_with(api_key="k", model_name="pm", temperature=0.2, ssl_verify=True)


def test_factory_openai_pronunciation_ignores_thinking_level():
    from pipeline.factory import create_pronunciation_generator
    cfg = MagicMock(pronunciation_engine="openai", pronunciation_model_name="gpt", pronunciation_temperature=0.2,
                    openai_api_key="k", openai_service_tier="auto", ssl_verify=True)
    with patch("pipeline.factory.OpenAIScriptGenerator") as oa:
        create_pronunciation_generator(cfg)
    assert "text_thinking_level" not in oa.call_args.kwargs and "thinking_level" not in oa.call_args.kwargs


def _thinking_cfg(engine, level_line):
    return _config(f"""
pronunciation:
  enabled: true
  engine: "{engine}"
  model_name: "m"
  temperature: 0.2
{level_line}""")


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_config_thinking_level_valid(level):
    assert _thinking_cfg("gemini", f'  thinking_level: "{level}"\n').pronunciation_thinking_level == level


def test_config_thinking_level_required_for_gemini():
    with pytest.raises(ValueError, match="thinking_level"):
        _thinking_cfg("gemini", "").pronunciation_thinking_level


@pytest.mark.parametrize("bad", ['"HIGH"', '"minimal"', "1", "true"])
def test_config_thinking_level_invalid(bad):
    with pytest.raises(ValueError, match="low, medium, high"):
        _thinking_cfg("gemini", f"  thinking_level: {bad}\n").pronunciation_thinking_level


def test_config_thinking_level_ignored_for_openai():
    assert _thinking_cfg("openai", "").pronunciation_thinking_level is None
    assert _thinking_cfg("openai", '  thinking_level: "bogus"\n').pronunciation_thinking_level is None


def test_real_config_thinking_level_high():
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        cfg = PipelineConfig("config.yaml")
    assert cfg.pronunciation_thinking_level == "high"


def _gemini_gen():
    from pipeline.generator import GeminiScriptGenerator
    with patch("pipeline.generator.genai.Client") as client_cls:
        gen = GeminiScriptGenerator(api_key="k", model_name="m", temperature=0.2)
    client = client_cls.return_value
    client.models.generate_content.return_value = MagicMock(text="out", usage_metadata=None)
    return gen, client


def test_gemini_generate_text_passes_thinking_level():
    from google.genai import types
    gen, client = _gemini_gen()
    with patch("pipeline.generator.log_gemini_usage"):
        assert gen.generate_text("sys", "user", thinking_level="low") == "out"
    config = client.models.generate_content.call_args.kwargs["config"]
    assert isinstance(config.thinking_config, types.ThinkingConfig)
    assert config.thinking_config.thinking_level.value.lower() == "low"
    assert config.temperature == 0.2 and config.system_instruction == "sys"


def test_gemini_generate_text_without_level_has_no_thinking_config():
    gen, client = _gemini_gen()
    with patch("pipeline.generator.log_gemini_usage"):
        gen.generate_text("sys", "user")
    assert client.models.generate_content.call_args.kwargs["config"].thinking_config is None


def test_gemini_batch_request_carries_thinking_level():
    from google.genai import types
    gen, client = _gemini_gen()
    gen.submit_batch("sys", "user", temperature=0.2, custom_id="c", thinking_level="high")
    request = client.batches.create.call_args.kwargs["src"][0]
    assert request["config"]["thinking_config"] == {"thinking_level": "high"}
    parsed = types.InlinedRequest(**request)  # the SDK accepts the request shape
    assert parsed.config.thinking_config.thinking_level.value.lower() == "high"
    gen.submit_batch("sys", "user", temperature=0.2, custom_id="c")
    assert "thinking_config" not in client.batches.create.call_args.kwargs["src"][0]["config"]


def test_gemini_other_calls_unaffected_by_thinking_level():
    gen, client = _gemini_gen()
    with patch("pipeline.generator.log_gemini_usage"):
        gen.polish_siman_script("draft", "rel", "instr")
    assert client.models.generate_content.call_count == 1
    for call in client.models.generate_content.call_args_list:
        assert getattr(call.kwargs.get("config"), "thinking_config", None) is None


def test_main_save_transcript_without_pronunciation_stage(tmp_path):
    import main as main_mod
    cfg = MagicMock(output_dir=str(tmp_path), section_slug="YD")
    main_mod.save_transcript(94, ORIG, None, cfg, "m")
    assert (tmp_path / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8") == ORIG
    assert not any("pronunciation" in n for n in os.listdir(tmp_path))


def test_main_save_transcript_with_changes_list(tmp_path):
    import main as main_mod
    cfg = MagicMock(output_dir=str(tmp_path), section_slug="YD")
    main_mod.save_transcript(94, VOCALIZED, CHANGES, cfg, "m")
    assert (tmp_path / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8") == VOCALIZED
    assert (tmp_path / "YD_Siman_94_m_pronunciation_changes.txt").read_text(encoding="utf-8") == format_changes(CHANGES)


def test_main_tts_failure_is_stage_error(tmp_path):
    import main as main_mod
    cfg = MagicMock(output_dir=str(tmp_path), section_slug="YD")
    args = MagicMock(skip_tts=False)
    with patch.object(main_mod, "create_tts_engine", side_effect=RuntimeError("quota")):
        with pytest.raises(main_mod.SimanStageError) as exc:
            main_mod.finish_siman(94, ORIG, None, cfg, args, "m")
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
        "  - Siman 95 | stage: Stage 3.5 (Pronunciation nikkud) | reason: boom"
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


def test_vocalize_script_errors_without_writing(tmp_path, monkeypatch, capsys):
    import scripts.vocalize_pronunciation as script
    monkeypatch.chdir(tmp_path)  # script chdirs to the repo root; restore cwd afterwards
    src = tmp_path / "x_transcript.txt"
    src.write_text(ORIG, encoding="utf-8")
    gen = MagicMock(model_name="m")
    gen.generate_text.return_value = VOCALIZED.replace("עוסק", "דן")
    monkeypatch.setattr("pipeline.config.PipelineConfig", MagicMock())
    monkeypatch.setattr("pipeline.pronunciation.create_pronunciation_vocalizer",
                        lambda c: PronunciationVocalizer(gen, "INSTR", None))
    with pytest.raises(SystemExit) as exc:
        script.main([str(src)])
    assert exc.value.code == 1
    assert "עוסק -> דן (letters changed" in capsys.readouterr().err
    assert sorted(os.listdir(tmp_path)) == ["x_transcript.txt"]


def test_vocalize_script_applies_dictionary_nikkud_first(tmp_path, monkeypatch):
    import scripts.vocalize_pronunciation as script
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "x.txt"
    src.write_text('סימן הא והש"ך', encoding="utf-8")
    gen = MagicMock(model_name="m")
    gen.generate_text.side_effect = lambda instr, text, thinking_level: text
    monkeypatch.setattr("pipeline.config.PipelineConfig", MagicMock())
    monkeypatch.setattr("pipeline.pronunciation.create_pronunciation_vocalizer",
                        lambda c: PronunciationVocalizer(gen, "INSTR", None))
    script.main([str(src)])
    assert gen.generate_text.call_args.args[1] == "סימן הֵי והשַׁךְ"
    assert (tmp_path / "x_vocalized.txt").read_text(encoding="utf-8") == "סימן הֵי והשַׁךְ"
    assert (tmp_path / "x_vocalized_changes.txt").read_text(encoding="utf-8") == ""


# --- scripts ----------------------------------------------------------------

def test_vocalize_script_args(tmp_path):
    from scripts.vocalize_pronunciation import parse_args, changes_path_for
    args = parse_args([str(tmp_path / "x_transcript.txt")])
    assert args.output == str(tmp_path / "x_transcript_vocalized.txt")
    assert changes_path_for(args.output) == str(tmp_path / "x_transcript_vocalized_changes.txt")


def test_tts_script_args(tmp_path):
    from scripts.tts_from_file import parse_args
    args = parse_args([str(tmp_path / "a.txt"), "--output", str(tmp_path / "a.mp3")])
    assert args.output == str(tmp_path / "a.mp3")
