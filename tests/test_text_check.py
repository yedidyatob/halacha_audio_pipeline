"""The text check (pipeline/text_check.py) with the repo config.yaml text_check.allowed_characters."""
import pytest

from pipeline.generator import BatchResultError
from pipeline.text_check import DisallowedCharactersError, check_allowed_characters, disallowed_characters_report
from tests.allowed_characters import ALLOWED_CHARACTERS

CLEAN = ('סימן פא זין, סעיף אלף: עֻבָּר בְּהֵמָה — הש"ך והט״ז (בדף 76) אומרים... "כך" ‘ו’ “זה” …\n'
         'מֵ־עַתָּה! a-z, 0-9; [x] {y} 50% \t\u00a0\u2013\u200f')


def test_clean_hebrew_lesson_passes():
    assert disallowed_characters_report(CLEAN, ALLOWED_CHARACTERS) is None
    assert check_allowed_characters(CLEAN, ALLOWED_CHARACTERS, "Siman 87 polish") == CLEAN


@pytest.mark.parametrize("text, code_point", [("הש\"ך מفרט את הדין", "U+0641"), ("שיעור дין", "U+0434")])
def test_arabic_or_cyrillic_letter_fails_with_code_point_and_context(text, code_point):
    with pytest.raises(DisallowedCharactersError) as exc:
        check_allowed_characters(text, ALLOWED_CHARACTERS, "Siman 87 polish")
    message = str(exc.value)
    assert message.startswith("Siman 87 polish: 1 character(s) outside text_check.allowed_characters: ")
    assert code_point in message and f"in '...{text}...'" in message


def test_report_quotes_five_characters_and_counts_the_rest():
    report = disallowed_characters_report("שלום " + "д" * 7, ALLOWED_CHARACTERS)
    assert report.startswith("7 character(s) outside text_check.allowed_characters: ")
    assert report.count("U+0434") == 5 and report.endswith(" (+2 more)")


def test_context_is_cut_around_the_character():
    text = "א" * 30 + "д" + "ב" * 30
    report = disallowed_characters_report(text, ALLOWED_CHARACTERS)
    assert f"in '...{'א' * 15}д{'ב' * 15}...'" in report


def test_error_is_a_batch_result_error():
    # a completed batch job with a bad character is resubmitted, not fetched again until --batch-timeout
    assert issubclass(DisallowedCharactersError, BatchResultError)
