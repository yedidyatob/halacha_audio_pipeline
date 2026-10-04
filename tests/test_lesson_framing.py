"""Fixed lesson intro/outro: structure- and placeholder-based tests, plus one exact-wording test
for the user-approved text (siman 94)."""
import os
import re
from unittest.mock import mock_open, patch

import pytest
import yaml

from pipeline.config import PipelineConfig
from pipeline.gematria import LETTER_NAMES, int_to_gematria, int_to_spoken_gematria
from pipeline.lesson_framing import (
    PLACEHOLDERS,
    render_lesson_frame,
    validate_lesson_template,
    wrap_lesson_with_frames,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINTS = re.compile("[\u0591-\u05C7]")


def _repo_framing():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["lesson_framing"]


def _load(yaml_text: str) -> PipelineConfig:
    base = 'halachic_section: "Yoreh De\'ah"\ntts:\n  chunk_gap_paragraph_ms: 700\n  chunk_gap_sentence_ms: 300\n'
    with patch("builtins.open", mock_open(read_data=base + yaml_text)), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        return PipelineConfig("config.yaml")


# --- rendering -------------------------------------------------------------

def test_all_placeholders_are_filled():
    out = render_lesson_frame("{hebrew_section}|{spoken_siman}|{gematria_siman}|{siman}", 94, "SEC")
    section, spoken, gem, num = out.split("|")
    assert section == "SEC"
    assert spoken == int_to_spoken_gematria(94)
    assert gem == int_to_gematria(94)
    assert num == "94"


def test_no_unfilled_braces_remain():
    template = " ".join("{" + p + "}" for p in PLACEHOLDERS)
    assert "{" not in render_lesson_frame(template, 95, "x") and "}" not in render_lesson_frame(template, 95, "x")


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_empty_template_renders_empty(empty):
    assert render_lesson_frame(empty, 94, "x") == ""


@pytest.mark.parametrize("bad", ["94-97", [94, 95], None, 94.0, True])
def test_siman_must_be_a_single_int(bad):
    with pytest.raises(TypeError, match="single siman"):
        render_lesson_frame("{siman}", bad, "x")


def test_wrap_order_and_separation():
    wrapped = wrap_lesson_with_frames("  BODY  ", 94, "SEC", "INTRO {siman}", "OUTRO {siman}")
    assert wrapped == "INTRO 94\n\nBODY\n\nOUTRO 94"


def test_wrap_skips_empty_parts():
    assert wrap_lesson_with_frames("BODY", 94, "x", "", "") == "BODY"
    assert wrap_lesson_with_frames("BODY", 94, "x", "I", "") == "I\n\nBODY"
    assert wrap_lesson_with_frames("", 94, "x", "I", "O") == "I\n\nO"


def test_each_siman_gets_its_own_frame():
    for siman in (94, 95, 96, 97):
        out = wrap_lesson_with_frames("B", siman, "x", "{siman}", "{spoken_siman}")
        assert out.startswith(str(siman)) and out.endswith(int_to_spoken_gematria(siman))


# --- template validation / config ------------------------------------------

def test_validate_rejects_unknown_placeholder():
    with pytest.raises(ValueError, match="nope"):
        validate_lesson_template("intro", "hello {nope}")


def test_validate_rejects_stray_brace():
    with pytest.raises(ValueError, match="lesson_framing.outro"):
        validate_lesson_template("outro", "hello {")


@pytest.mark.parametrize("missing", ["intro", "outro"])
def test_config_requires_both_framing_keys(missing):
    keys = {"intro": '"a"', "outro": '"b"'}
    del keys[missing]
    text = "lesson_framing:\n" + "".join(f"  {k}: {v}\n" for k, v in keys.items())
    with pytest.raises(ValueError, match=missing):
        _load(text)


def test_config_requires_framing_section():
    with pytest.raises(ValueError, match="lesson_framing"):
        _load("")


def test_config_rejects_unknown_placeholder():
    with pytest.raises(ValueError, match="bogus"):
        _load('lesson_framing:\n  intro: "x {bogus}"\n  outro: "y"\n')


def test_config_loads_templates_stripped_and_empty_allowed():
    cfg = _load('lesson_framing:\n  intro: "  hi {siman}  "\n  outro: ""\n')
    assert cfg.lesson_intro_template == "hi {siman}"
    assert cfg.lesson_outro_template == ""


# --- the real config.yaml ---------------------------------------------------

def test_repo_config_has_valid_framing():
    framing = _repo_framing()
    for key in ("intro", "outro"):
        assert isinstance(framing[key], str) and framing[key].strip()
        validate_lesson_template(key, framing[key])


def test_repo_intro_and_outro_are_short_and_use_the_siman():
    framing = _repo_framing()
    assert "{spoken_siman}" in framing["intro"] and "{hebrew_section}" in framing["intro"]
    assert "{spoken_siman}" in framing["outro"]
    assert len(framing["intro"].strip().splitlines()) <= 5
    assert len(framing["outro"].strip().splitlines()) <= 5


def test_repo_framing_renders_for_every_siman_without_gaps():
    framing = _repo_framing()
    for siman in range(1, 404):
        for key in ("intro", "outro"):
            out = render_lesson_frame(framing[key], siman, "יורה דעה")
            assert out and "{" not in out and "}" not in out
            assert int_to_spoken_gematria(siman) in out or key == "intro"


def test_repo_config_hebrew_is_not_mojibake():
    """config.yaml must stay valid UTF-8 Hebrew (no double-encoded 'ל×' style text)."""
    with open(os.path.join(ROOT, "config.yaml"), "rb") as f:
        text = f.read().decode("utf-8")
    assert not re.search("[\u00d7\u00d6\u00c3\u00e2][\u0080-\u00bf\u2000-\u20ff\u02c6\u0152-\u0192]", text)
    assert len(re.findall("[\u05d0-\u05ea]", text)) > 1000


def test_polishing_instruction_forbids_llm_greeting_and_closing():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f)
    instruction = data["polishing_instruction"]
    assert "בקוד" in instruction  # tells the model the frame is added in code
    assert "פתח וסיים בברכות" not in instruction  # the old "open and close with blessings" rule is gone


# The user-approved wording (kept verbatim, only the siman is a placeholder).
EXPECTED_INTRO_94 = (
    "שלום לכולם, נפתח את סימן צָדִי דלת ביורה דעה.\n"
    "זה שיעור הכנה למבחני הרבנות. נתחיל מהטור והבית יוסף, נמשיך לשולחן ערוך והרמ\"א, ונסיים בנושאי הכלים.\n"
    "אחרי כל חלק, סיכום קצר של עיקרי הדברים.\n"
    "נתחיל."
)
EXPECTED_OUTRO_94 = (
    "עד כאן סימן צָדִי דלת.\n"
    "מומלץ לעצור כאן, לחזור על הדברים בעל פה, ורק אז להמשיך.\n"
    "תודה שהקשבתם, והצלחה רבה בהכנה למבחנים.\n"
    "נתראה בשיעור הבא, בעזרת השם."
)


def test_repo_framing_renders_the_approved_wording_for_siman_94():
    framing = _repo_framing()
    assert render_lesson_frame(framing["intro"], 94, "יורה דעה") == EXPECTED_INTRO_94
    assert render_lesson_frame(framing["outro"], 94, "יורה דעה") == EXPECTED_OUTRO_94


def test_repo_framing_uses_hebrew_section_without_a_doubled_prefix():
    # {hebrew_section} yields the bare name (e.g. "יורה דעה"); the template supplies the ב prefix.
    intro = _repo_framing()["intro"]
    assert "ב{hebrew_section}" in intro
    assert "באורח חיים" in render_lesson_frame(intro, 94, "אורח חיים")


def test_repo_framing_has_no_mojibake_and_nikkud_only_in_the_spoken_siman():
    framing = _repo_framing()
    for key in ("intro", "outro"):
        text = render_lesson_frame(framing[key], 94, "יורה דעה")
        text = text.replace(int_to_spoken_gematria(94), "")  # צָדִי is voweled on purpose (TTS)
        assert "\ufffd" not in text and "\u00d7" not in text and "\u00c3" not in text
        assert not POINTS.search(text)  # the rest of the fixed text is plain, unvoweled Hebrew
