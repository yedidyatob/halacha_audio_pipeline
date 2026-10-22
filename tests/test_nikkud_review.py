import os
from unittest.mock import MagicMock, patch, mock_open

import pytest

from pipeline.nikkud_review import (
    strip_nikkud, has_nikkud, vocalized_words, format_words, format_review, VocalizedWord,
    WORDS_HEADER, OVERRIDES_TITLE, OVERRIDES_HEADER,
)
from pipeline.apply_nikkud import apply_nikkud_with_overrides, NikkudOverride

# A final transcript: the polish vocalized טֵית / פַּת / סוֹפֵר, the dictionary wrote השַׁךְ.
VOCALIZED = "סימן טֵית עוסק בדין פַּת.\nזו טעות סוֹפֵר בדברי השַׁךְ.\n"


# --- nikkud helpers -----------------------------------------------------------

def test_strip_nikkud_keeps_letters_and_punctuation():
    assert strip_nikkud(VOCALIZED) == "סימן טית עוסק בדין פת.\nזו טעות סופר בדברי השך.\n"
    assert strip_nikkud("בית־דין׀") == "בית־דין׀"  # maqaf / paseq are not nikkud
    assert has_nikkud("טֵית") and not has_nikkud("טית")


# --- review file ------------------------------------------------------------

def test_vocalized_words_line_numbers_and_context():
    words = " ".join(f"w{i}" for i in range(12))
    [word] = vocalized_words(f"\n\nאחד שתיים\n{words} כְּחָל {words}\n")
    assert word.line == 4  # leading blank lines counted, as in the saved file
    assert word.context == "...w7 w8 w9 w10 w11 [כְּחָל] w0 w1 w2 w3 w4..."
    assert vocalized_words("(כְּחָל), x") == [VocalizedWord("כחל", "כְּחָל", 1, "([כְּחָל]), x")]
    assert [w.vocalized for w in vocalized_words(VOCALIZED)] == ["טֵית", "פַּת", "סוֹפֵר", "השַׁךְ"]


def test_format_words_groups_flags_variants_and_sorts():
    text = ("מה דינו של כְּחָל של בהמה, ופַת.\n"
            "אם כֹּחַל נאסר ופַת אחרת.\n"
            "ואכל פַּת ובשר כְּחָל, ופַת.\n"
            "וכּוּתָח שלם.")
    assert format_words(vocalized_words(text)).splitlines() == [
        "varies\tplain\tvocalized\tcount\tfirst_line\tcontext",
        # כחל got two readings: first, each variant on its own line, the more frequent first
        "yes\tכחל\tכְּחָל\t2\t1\tמה דינו של [כְּחָל] של בהמה, ופַת.",
        "yes\tכחל\tכֹּחַל\t1\t2\tאם [כֹּחַל] נאסר ופַת אחרת.",
        # then the rest by count (ties: first line)
        "no\tופת\tופַת\t3\t1\t...דינו של כְּחָל של בהמה, [ופַת].",   # 5 words each side, "..." when cut
        "no\tפת\tפַּת\t1\t3\tואכל [פַּת] ובשר כְּחָל, ופַת.",
        "no\tוכותח\tוכּוּתָח\t1\t4\t[וכּוּתָח] שלם.",
    ]


def test_format_review_lists_words_then_dictionary_overrides():
    # the polish vocalized כחל and the Rambam its own way; the dictionary overrides both
    final, overrides = apply_nikkud_with_overrides('דין פַּת.\nאמר הָרַמְבַּ"ם שאסור בכֹּחַל של בהמה.')
    lines = format_review(final, overrides).splitlines()
    rambam = lines.index(OVERRIDES_HEADER) + 1
    assert lines[0] == WORDS_HEADER
    assert lines[rambam - 3] == "" and lines[rambam - 2] == OVERRIDES_TITLE
    assert lines[rambam].startswith('הָרַמְבַּ"ם\t' + overrides[0].dictionary + "\t2\tאמר [")
    assert lines[rambam + 1].split("\t")[:3] == ["בכֹּחַל", overrides[1].dictionary, "2"]
    assert len(lines) == rambam + 2
    # the words section is the final transcript's: the dictionary's forms, not the model's
    words = lines[1:rambam - 3]
    assert any(overrides[0].dictionary in row for row in words)
    assert not any("הָרַמְבַּ\"ם" in row for row in words)


def test_format_review_without_vocalization_is_the_headers():
    assert format_review("טקסט", []) == "\n".join([WORDS_HEADER, "", OVERRIDES_TITLE, OVERRIDES_HEADER]) + "\n"


def test_override_context_is_cut_at_five_words():
    text = "a b c d e f כְּחָל g h i j k l"
    override = NikkudOverride("כֹּחַל", "כְּחָל", text.index("כְּחָל"))
    last = format_review(text, [override]).splitlines()[-1]
    assert last == "כֹּחַל\tכְּחָל\t1\t...b c d e f [כְּחָל] g h i j k..."


# --- dictionary nikkud stage ---------------------------------------------------

def test_dictionary_respells_letter_name_he_in_letter_context():
    assert apply_nikkud_with_overrides("סימן הא")[0] == "סימן הֵי"
    assert apply_nikkud_with_overrides("סימן הֵא")[0] == "סימן הֵי"  # the model's misvocalization too
    assert apply_nikkud_with_overrides("הא דתניא")[0] == "הא דתניא"  # not a letter name


# --- config properties -------------------------------------------------------

BASE_YAML = """
halachic_section: "Yoreh De'ah"
sefaria:
  base_url: "https://www.sefaria.org/api"
  timeout_seconds: 15
  retries: 3
  ssl_verify: true
directories:
  thinking_dir: "./output/thinking"
generator:
  engine: "gemini"
tts:
  engine: "elevenlabs"
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


def _placeholder_config(always, never, prompt="תמיד: {always_vocalize} | לעולם: {never_vocalize} | {tts_abbreviations}"):
    return _config(f"""
polishing_instruction: '{prompt}'
pronunciation:
  always_vocalize: {always}
  never_vocalize: {never}
""")


def test_config_missing_section_errors():
    cfg = _config('polishing_instruction: "x"\n')
    for prop in ("pronunciation_always_vocalize", "pronunciation_never_vocalize", "polishing_instruction"):
        with pytest.raises(ValueError, match="pronunciation"):
            getattr(cfg, prop)


def test_polishing_instruction_required():
    with pytest.raises(ValueError, match="polishing_instruction"):
        _config("pronunciation:\n  always_vocalize: []\n  never_vocalize: []\n").polishing_instruction


def test_instruction_placeholders_filled():
    cfg = _placeholder_config('["עובר בהמה", "כותח"]', '["דאורייתא", "דרבנן"]')
    assert cfg.pronunciation_always_vocalize == ["עובר בהמה", "כותח"]
    assert cfg.pronunciation_never_vocalize == ["דאורייתא", "דרבנן"]
    assert cfg.polishing_instruction.startswith("תמיד: עובר בהמה, כותח | לעולם: דאורייתא, דרבנן | ")


def test_instruction_empty_lists_render_none_word():
    cfg = _placeholder_config("[]", "[]")
    assert cfg.polishing_instruction.startswith("תמיד: אין | לעולם: אין | ")


def test_instruction_missing_lists_error():
    cfg = _config("""
polishing_instruction: "x {always_vocalize}"
pronunciation:
  never_vocalize: []
""")
    with pytest.raises(ValueError, match="always_vocalize"):
        cfg.polishing_instruction


KAHAL = '{word: "כחל", readings: [{nikkud: "כְּחָל", meaning: "עטין של בהמה"}, {nikkud: "כֹּחַל", meaning: "צבע איפור"}]}'


def test_always_vocalize_mixes_plain_words_and_readings():
    single = '{word: "עובר", readings: [{nikkud: "עֻבָּר", meaning: "ולד בבטן"}]}'   # one reading: a hint
    cfg = _placeholder_config(f'["כותח", {KAHAL}, {single}]', '["דרבנן"]')
    assert cfg.pronunciation_always_vocalize == [
        "כותח", "כחל (כְּחָל = עטין של בהמה; כֹּחַל = צבע איפור)", "עובר (עֻבָּר = ולד בבטן)"]
    assert cfg.polishing_instruction.startswith(
        "תמיד: כותח, כחל (כְּחָל = עטין של בהמה; כֹּחַל = צבע איפור), עובר (עֻבָּר = ולד בבטן) | לעולם: דרבנן | ")


def test_reading_may_add_matres_like_the_validator():
    cfg = _placeholder_config('[{word: "הקילו", readings: [{nikkud: "הֵקֵלּוּ", meaning: "הקלו"}]}]', "[]")
    assert cfg.pronunciation_always_vocalize == ["הקילו (הֵקֵלּוּ = הקלו)"]


@pytest.mark.parametrize("entry, message", [
    ('{readings: [{nikkud: "כְּחָל", meaning: "m"}]}', "'word' and 'readings'"),
    ('{word: "כחל", readings: [{nikkud: "כְּחָל", meaning: "m"}], note: "x"}', "'word' and 'readings'"),
    ('{word: "", readings: [{nikkud: "כְּחָל", meaning: "m"}]}', "non-empty word without nikkud"),
    ('{word: "כְּחָל", readings: [{nikkud: "כְּחָל", meaning: "m"}]}', "non-empty word without nikkud"),
    ('{word: "כחל", readings: []}', "non-empty list"),
    ('{word: "כחל", readings: "כְּחָל"}', "non-empty list"),
    ('{word: "כחל", readings: [{nikkud: "כְּחָל"}]}', "'nikkud' and 'meaning'"),
    ('{word: "כחל", readings: [{nikkud: "כְּחָל", meaning: " "}]}', "non-empty 'meaning'"),
    ('{word: "כחל", readings: [{nikkud: "", meaning: "m"}]}', "must be the vocalized word"),
    ('{word: "כחל", readings: [{nikkud: "כחל", meaning: "m"}]}', "must be the vocalized word"),
    ('{word: "כחל", readings: [{nikkud: "כְּחָלִים", meaning: "m"}]}', "letters changed"),          # typo
    ('{word: "כחל", readings: [{nikkud: "וְכְּחָל", meaning: "m"}]}', "at the start of the word"),
    ('{word: "ולד", readings: [{nikkud: "לָד", meaning: "m"}]}', "at the start of the word"),     # first letter
])
def test_always_vocalize_reading_errors(entry, message):
    cfg = _placeholder_config(f"[{entry}]", "[]")
    with pytest.raises(ValueError, match=message):
        cfg.polishing_instruction


@pytest.mark.parametrize("bad", ['"כותח"', '[1, 2]', '[""]'])
def test_instruction_invalid_list_errors(bad):
    cfg = _placeholder_config(bad, "[]")
    with pytest.raises(ValueError, match="always_vocalize"):
        cfg.pronunciation_always_vocalize


def test_real_config_yaml_combined_prompt():
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        cfg = PipelineConfig("config.yaml")
    instruction = cfg.polishing_instruction
    assert "{" not in instruction and "}" not in instruction
    # the polish instructions, then the nikkud addendum
    assert instruction.index("אתה משמש כרב מורה הוראה") < instruction.index("===== תוספת: ניקוד לצורך הגייה =====")
    assert "תמיד לנקד (כשמופיעות קריאות בסוגריים, בחר לפי ההקשר): עובר בהמה, כותח" in instruction
    assert ('גזרה (גְּזֵרָה = צו או תקנה, כמו "גזרה שווה"; גִּזְרָה = חיתוך, צורה)'
            in instruction)
    assert "כחל" not in instruction.split("תמיד לנקד")[1].split("\n")[0]  # in the nikkud dictionary instead
    assert ('למה (לְמָה = אל מה, לאיזו כוונה, כמו "למה התכוון"; לָמָּה = מדוע, כמו "למה צריך"), שה'
            in instruction)  # readings keep the entry's position
    assert "לעולם לא לנקד: דאורייתא, דרבנן" in instruction
    assert "אתה עורך הגייה" not in instruction  # the separate stage's role paragraph is gone

# The nikkud section of the rendered prompt, pinned byte for byte (it started as #20's combined
# prompt; the "already vocalized", letter-name bullet and letter-name examples were since removed).
NIKKUD_SECTION = '===== תוספת: ניקוד לצורך הגייה =====\nכל ההנחיות לעיל לעריכת השיעור נשארות בתוקף. בנוסף, בתסריט המלוטש שאתה כותב,\nהוסף ניקוד רק למילים שמנוע ה-TTS עלול לקרוא לא נכון, בלי שום תגיות או סימון.\nהכללים לבחירת המילים:\nנקד מילה רק אם קורא עברית בלי ניקוד עלול לקרוא אותה בשתי דרכים שונות, ובהקשר הזה נכונה דווקא אחת מהן:\n- אותן אותיות, משמעות והגייה אחרת: עובר (עֻבָּר / עוֹבֵר), פת, ספר, סופר, מבשר, כלי.\n- מילה שנוטים לקרוא בטעות: כותח, נבילה, הקילו.\nאל תנקד מילה רק בגלל שהיא ארמית או תורנית, אם יש לה קריאה מקובלת אחת (דאורייתא, דרבנן, מותר, אסור, בדיעבד).\nאם אינך בטוח, אל תנקד.\n\nתמיד לנקד (כשמופיעות קריאות בסוגריים, בחר לפי ההקשר): עובר בהמה, כותח, נבילה, למה (לְמָה = אל מה, לאיזו כוונה, כמו "למה התכוון"; לָמָּה = מדוע, כמו "למה צריך"), שה, גזרה (גְּזֵרָה = צו או תקנה, כמו "גזרה שווה"; גִּזְרָה = חיתוך, צורה)\nלעולם לא לנקד: דאורייתא, דרבנן\n\nנקד רק את המילה הדו-משמעית עצמה (כולל אותיות השימוש הצמודות לה), בניקוד מלא ומדויק לפי ההקשר, ואל תנקד את שאר המשפט.\n\nדוגמאות:\nעובר בהמה ← עֻבָּר בְּהֵמָה\nכותח ← כּוּתָח\nאכל פת ← אכל פַּת\nטעות סופר ← טעות סוֹפֵר\n'


def test_real_config_nikkud_section_is_pinned():
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        instruction = PipelineConfig("config.yaml").polishing_instruction
    assert instruction[instruction.index("===== תוספת"):] == NIKKUD_SECTION


def _gemini_gen():
    from pipeline.generator import GeminiScriptGenerator
    from tests.gemini_fakes import make_gemini_generator, answer_stream
    gen, client = make_gemini_generator("unused", model_name="m", temperature=0.2)
    client.models.generate_content_stream.side_effect = lambda **_: answer_stream("out", thought="")
    return gen, client


def test_gemini_generate_text_passes_thinking_level():
    from google.genai import types
    gen, client = _gemini_gen()
    with patch("pipeline.generator.log_gemini_usage"):
        assert gen.generate_text("sys", "user", thinking_level="low") == "out"
    config = client.models.generate_content_stream.call_args.kwargs["config"]
    assert isinstance(config.thinking_config, types.ThinkingConfig)
    assert config.thinking_config.thinking_level.value.lower() == "low"
    assert config.temperature == 0.2 and config.system_instruction == "sys"


def test_gemini_generate_text_without_level_sends_no_thinking_level():
    gen, client = _gemini_gen()
    with patch("pipeline.generator.log_gemini_usage"):
        gen.generate_text("sys", "user")
    thinking = client.models.generate_content_stream.call_args.kwargs["config"].thinking_config
    assert thinking.thinking_level is None and thinking.include_thoughts is True


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


def test_gemini_polish_sends_no_thinking_level():
    gen, client = _gemini_gen()
    with patch("pipeline.generator.log_gemini_usage"):
        gen.polish_siman_script("draft", "rel", "instr", temperature=0.1)
    assert client.models.generate_content_stream.call_count == 1
    for call in client.models.generate_content_stream.call_args_list:
        assert call.kwargs["config"].thinking_config.thinking_level is None


def test_main_save_transcript_writes_transcript_and_review(tmp_path):
    import main as main_mod
    cfg = MagicMock(output_dir=str(tmp_path), section_slug="YD")
    final, overrides = apply_nikkud_with_overrides('זו טעות סוֹפֵר בדברי הַשַׁ"ךְ.')
    main_mod.save_transcript(94, final, overrides, cfg, "m")
    assert (tmp_path / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8") == final
    assert ((tmp_path / "YD_Siman_94_m_pronunciation_changes.txt").read_text(encoding="utf-8")
            == format_review(final, overrides))
    assert overrides


def test_main_tts_failure_is_stage_error(tmp_path):
    import main as main_mod
    cfg = MagicMock(output_dir=str(tmp_path), section_slug="YD")
    args = MagicMock(skip_tts=False)
    with patch.object(main_mod, "create_tts_engine", side_effect=RuntimeError("quota")):
        with pytest.raises(main_mod.SimanStageError) as exc:
            main_mod.finish_siman(94, VOCALIZED, [], cfg, args, "m")
    assert exc.value.stage == main_mod.STAGE_TTS and exc.value.reason == "quota"


def test_run_summary_format():
    import main as main_mod
    summary = main_mod.RunSummary()
    summary.success(94)
    summary.failure(95, main_mod.STAGE_NIKKUD, "boom")
    summary.success(96)
    assert summary.has_failures
    assert summary.format() == (
        "===== Run summary =====\n"
        "Succeeded (2/3): 94, 96\n"
        "Failed (1/3): 95\n"
        "  - Siman 95 | stage: Stage 3.1 (Dictionary nikkud) | reason: boom"
    )
    empty = main_mod.RunSummary()
    empty.success(1)
    assert not empty.has_failures
    assert "Failed (0/1): none" in empty.format()


def _run_main_sync(tmp_path, monkeypatch, polish):
    """Runs main.main() in sync mode for simanim 94,95 with cached drafts and relations."""
    import main as main_mod
    cfg = MagicMock(section_slug="YD", generator_max_retries=0, drafts_dir=str(tmp_path), relations_dir=str(tmp_path),
                    output_dir=str(tmp_path))
    for s in (94, 95):
        (tmp_path / f"YD_Siman_{s}_m_draft.txt").write_text(f"draft {s}", encoding="utf-8")
    (tmp_path / "YD_Relations_94_95_m.txt").write_text("rel", encoding="utf-8")
    gen = MagicMock(model_name="m")
    gen.polish_siman_script.side_effect = polish
    monkeypatch.setattr("sys.argv", ["main.py", "94,95", "--skip-tts"])
    with patch.object(main_mod, "PipelineConfig", return_value=cfg), \
         patch.object(main_mod, "create_generator_engine", return_value=gen), \
         patch.object(main_mod, "SefariaExtractor"):
        main_mod.main()


def test_main_continues_after_siman_failure_and_exits_nonzero(tmp_path, monkeypatch, caplog):
    def polish(draft, rel, instr, temperature, thinking_name):
        if draft == "draft 94":
            raise RuntimeError("LLM down")
        return "polished"
    with pytest.raises(SystemExit) as exc:
        _run_main_sync(tmp_path, monkeypatch, polish)
    assert exc.value.code == 1
    assert (tmp_path / "YD_Siman_95_m_transcript.txt").exists()
    assert not (tmp_path / "YD_Siman_94_m_transcript.txt").exists()
    assert "Succeeded (1/2): 95" in caplog.text
    assert "Siman 94 | stage: Stage 3 (Polishing and nikkud) | reason: LLM down" in caplog.text


def test_main_all_succeed_exits_zero(tmp_path, monkeypatch, caplog):
    _run_main_sync(tmp_path, monkeypatch, lambda d, r, i, temperature, thinking_name: "polished")
    assert "Succeeded (2/2): 94, 95" in caplog.text
    for s in (94, 95):
        assert (tmp_path / f"YD_Siman_{s}_m_pronunciation_changes.txt").exists()


# --- scripts ----------------------------------------------------------------

def test_tts_script_args(tmp_path):
    from scripts.tts_from_file import parse_args
    args = parse_args([str(tmp_path / "a.txt"), "--output", str(tmp_path / "a.mp3")])
    assert args.output == str(tmp_path / "a.mp3")
