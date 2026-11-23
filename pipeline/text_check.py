"""
The text checks on a polished lesson: characters outside config text_check.allowed_characters (e.g. a stray
Arabic or Cyrillic letter), which fail the polish attempt, and sentences longer than
text_check.max_sentence_words, which are only reported.
"""
import re
from typing import Optional

from pipeline.generator import BatchResultError

SHOWN_CHARACTERS = 5  # bad characters quoted with their context in a report; the rest are counted
CONTEXT_CHARS = 15    # characters of context on each side of a quoted bad character


class DisallowedCharactersError(BatchResultError):
    """
    A model response contains characters outside text_check.allowed_characters. Like a blocked or empty
    response, a new call may fix it: sync mode retries, batch mode resubmits (generator.max_retries).
    """


def disallowed_characters_report(text: str, allowed_characters: str) -> Optional[str]:
    """None if every character of text is allowed; otherwise a report naming each bad character with context."""
    bad = [(m.group(), m.start()) for m in re.finditer(f"[^{allowed_characters}]", text)]
    if not bad:
        return None
    shown = "; ".join(f"'{c}' U+{ord(c):04X} in '...{text[max(0, i - CONTEXT_CHARS):i + CONTEXT_CHARS + 1]}...'"
                      for c, i in bad[:SHOWN_CHARACTERS])
    more = f" (+{len(bad) - SHOWN_CHARACTERS} more)" if len(bad) > SHOWN_CHARACTERS else ""
    return f"{len(bad)} character(s) outside text_check.allowed_characters: {shown}{more}"


def check_allowed_characters(text: str, allowed_characters: str, label: str) -> str:
    """Returns text unchanged, or raises DisallowedCharactersError with the report."""
    report = disallowed_characters_report(text, allowed_characters)
    if report:
        raise DisallowedCharactersError(f"{label}: {report}")
    return text


# --- Long sentences (text_check.max_sentence_words): a warning only, never a failed polish ---

SHOWN_SENTENCES = 3  # longest sentences quoted in a long-sentence report; the rest are counted

# A sentence ends at . ? ! or : before whitespace, or at a line break; an ellipsis ("..." or "…") is a
# pause inside the sentence, not its end.
SENTENCE_END = re.compile(r"(?<=[.?!:])(?<!\.\.\.)\s+")
NIKKUD = re.compile(r"[\u0591-\u05C7]")
# A spelled-out Gemara page reference: "דף" (or "בדף", "ודף", "ובדף"), the page number in letter names,
# and an optional side ("עמוד אלף", "עמוד אלף ועמוד בית"), e.g. "בדף קוף טית וו עמוד בית".
LETTER_NAME = r"(?:אלף|בית|גימל|דלת|הי|וו|זין|חית|טית|יוד|כף|למד|מם|נון|סמך|עין|פא|צדי|קוף|ריש|שין|תו)"
WHOLE_LETTER_NAME = rf"{LETTER_NAME}(?![^\W\d_])"  # not the start of a longer word ("בית" in "ביתא")
PAGE_REFERENCE = re.compile(rf"(?<!\S)ו?ב?דף(?:\s+{WHOLE_LETTER_NAME})+(?:\s+ו?עמוד\s+{WHOLE_LETTER_NAME})*")


def word_count(text: str) -> int:
    """Whitespace-separated tokens with a letter or digit (a dash alone is not a word)."""
    return sum(1 for token in text.split() if re.search(r"\w", token))


def split_sentences(text: str) -> list:
    """The non-empty sentences of text, in order."""
    return [sentence for line in text.splitlines() for sentence in SENTENCE_END.split(line.strip())
            if word_count(sentence)]


def words_without_page_references(sentence: str) -> int:
    """The sentence's word count, not counting its spelled-out Gemara page references (nikkud ignored)."""
    return word_count(PAGE_REFERENCE.sub(" ", NIKKUD.sub("", sentence)))


def long_sentences_report(text: str, max_words: int) -> Optional[str]:
    """
    None if no sentence has more than max_words words (page references not counted); otherwise a report
    of how many do, quoting the SHOWN_SENTENCES longest with both word counts.
    """
    counted = ((words_without_page_references(s), word_count(s), s) for s in split_sentences(text))
    long = sorted((item for item in counted if item[0] > max_words), key=lambda item: (-item[0], -item[1]))
    if not long:
        return None
    shown = "; ".join(f"[{without} words, {total} with page references] '{sentence}'"
                      for without, total, sentence in long[:SHOWN_SENTENCES])
    more = f" (+{len(long) - SHOWN_SENTENCES} more)" if len(long) > SHOWN_SENTENCES else ""
    return (f"{len(long)} sentence(s) over text_check.max_sentence_words ({max_words} words, not counting "
            f"Gemara page references): {shown}{more}")
