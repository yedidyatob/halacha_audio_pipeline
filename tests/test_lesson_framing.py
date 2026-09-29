import pytest
from pipeline.gematria import int_to_spoken_gematria
from pipeline.lesson_framing import render_lesson_frame, wrap_lesson_with_frames


def test_spoken_gematria_basic():
    assert int_to_spoken_gematria(94) == "×¦×“×™×§ ×“×œ×ª"
    assert int_to_spoken_gematria(15) == "×˜×™×ª ×•×•"
    assert int_to_spoken_gematria(1) == "××œ×£"


def test_render_and_wrap():
    intro = "×‘×¨×•×›×™× ×”×‘××™× ×œ{hebrew_section}, ×¡×™×ž×Ÿ {spoken_siman}."
    outro = "×¢×“ ×›××Ÿ ×¡×™×ž×Ÿ {gematria_siman}."
    body = "× ×ª×—×™×œ ×‘×¡×¢×™×£ ××œ×£."
    wrapped = wrap_lesson_with_frames(
        body=body,
        siman=94,
        hebrew_section="×™×•×¨×” ×“×¢×”",
        intro_template=intro,
        outro_template=outro,
    )
    assert wrapped.startswith("×‘×¨×•×›×™× ×”×‘××™× ×œ×™×•×¨×” ×“×¢×”, ×¡×™×ž×Ÿ ×¦×“×™×§ ×“×œ×ª.")
    assert "× ×ª×—×™×œ ×‘×¡×¢×™×£ ××œ×£." in wrapped
    assert wrapped.endswith('×¢×“ ×›××Ÿ ×¡×™×ž×Ÿ ×¦"×“.')


def test_empty_templates_passthrough():
    assert wrap_lesson_with_frames("×’×•×£", 94, "×™×•×¨×” ×“×¢×”", "", "") == "×’×•×£"
