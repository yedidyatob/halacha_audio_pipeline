"""The text checks (pipeline/text_check.py) with the repo config.yaml text_check settings."""
import pytest

from pipeline.generator import BatchResultError
from pipeline.text_check import (DisallowedCharactersError, check_allowed_characters, disallowed_characters_report,
                                 long_sentences_report, split_sentences, words_without_page_references)
from tests.text_check_config import ALLOWED_CHARACTERS, MAX_SENTENCE_WORDS

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


def _sentence(words: int, word: str = "מילה") -> str:
    return " ".join([word] * words) + "."


def test_sentences_end_at_punctuation_and_line_breaks_but_not_at_an_ellipsis():
    text = "רגע... זה לא סותר? לא! הנה הדין: מותר.\nשורה חדשה … בלי נקודה\n\n — \n"
    assert split_sentences(text) == ["רגע... זה לא סותר?", "לא!", "הנה הדין:", "מותר.", "שורה חדשה … בלי נקודה"]


@pytest.mark.parametrize("sentence, words", [
    ("כמבואר בחולין דף קוף טית וו עמוד בית, ולכן אסור.", 4),
    ("בעבודה זרה בדף מם טית עמוד אלף ובדף סמך חית עמוד אלף ועמוד בית שנינו.", 3),
    ("ודף קוֹף יוד דָּלֶת עמוד אָלֶף — כך.", 1),  # nikkud on the letter names
    ("הגמרא בדף ביתא אומרת.", 4),                   # "ביתא" is not the letter name בית
    ("הש\"ך — והט\"ז — אומרים.", 3),             # a dash alone is not a word
])
def test_page_references_are_not_counted(sentence, words):
    assert words_without_page_references(sentence) == words


def test_no_report_at_or_under_the_limit():
    assert long_sentences_report(_sentence(MAX_SENTENCE_WORDS) + "\n" + _sentence(3), MAX_SENTENCE_WORDS) is None
    # a page reference does not push a sentence over the limit
    assert long_sentences_report(_sentence(20) + " בדף קוף טית וו עמוד בית.", 20) is None


def test_report_quotes_the_three_longest_with_both_counts_and_counts_the_rest():
    text = "\n".join([_sentence(30, "א"), _sentence(26, "ב") + " בדף קוף עמוד בית.", _sentence(40, "ג"),
                     _sentence(27, "ד"), _sentence(10, "ה")])
    report = long_sentences_report(text, 25)
    assert report.startswith("4 sentence(s) over text_check.max_sentence_words (25 words, not counting "
                             "Gemara page references): [40 words, 40 with page references] 'ג ")
    assert report.index("[30 words, 30 with page references] 'א ") < report.index("[27 words, 27 with page references]")
    assert "ב ב" not in report and report.endswith(" (+1 more)")


def test_long_sentences_are_reported_only_once_over_the_limit():
    report = long_sentences_report(_sentence(26) + " " + _sentence(5), 25)
    assert report.startswith("1 sentence(s) over") and "[26 words, 26 with page references]" in report
    assert not report.endswith("more)")
