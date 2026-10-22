"""
Nikkud helpers and the nikkud review file of a lesson.

The Stage 3 polish vocalizes the words a TTS engine could read in more than one way, and the
dictionary stage (pipeline/apply_nikkud.py) then applies its fixed nikkud, overriding the
model's for every dictionary word. The review file (format_review) lists, for the final
transcript, every vocalized word with its context (words vocalized more than one way first),
then every place where the dictionary overrode the model's nikkud.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import NamedTuple

# Hebrew points and cantillation marks. Excludes maqaf (U+05BE), paseq (U+05C0),
# sof pasuq (U+05C3) and nun hafukha (U+05C6), which are punctuation, not nikkud.
_NIKKUD_RE = re.compile("[\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]")

# A ו / י that standard voweled spelling may add or remove (matres lectionis): one that follows
# a Hebrew letter, or a geresh / gershayim that follows one (abbreviations), so never the first
# letter of a word (e.g. the conjunction ו).
_MATRES_RE = re.compile("(?<=[\u05D0-\u05EA])[\u05D5\u05D9]|(?<=[\u05D0-\u05EA][\"'\u05F3\u05F4])[\u05D5\u05D9]")
_ANY_VAV_YOD_RE = re.compile("[\u05D5\u05D9]")

# Punctuation trimmed from the ends of a word in the review file.
_EDGE_PUNCT = ".,;:!?()[]{}-\u05BE\u2013\u2014"

# Words of context on each side of a listed word in the review file.
CONTEXT_WORDS = 5


def strip_nikkud(text: str) -> str:
    """Removes Hebrew nikkud and cantillation marks, keeping letters and punctuation."""
    return _NIKKUD_RE.sub("", text)


def has_nikkud(text: str) -> bool:
    return bool(_NIKKUD_RE.search(text))


def word_vocalization_error(word: str, vocalized: str) -> str | None:
    """
    Why `vocalized` is not a vocalization of the plain `word`, or None if it is: with nikkud
    removed it must be the word's letters, except that ו / י may be added or removed anywhere but
    at the first letter (עובר -> עֻבָּר, הקילו -> הֵקֵלּוּ).
    """
    letters = strip_nikkud(vocalized)
    if _MATRES_RE.sub("", letters) == _MATRES_RE.sub("", word):
        return None
    if _ANY_VAV_YOD_RE.sub("", letters) == _ANY_VAV_YOD_RE.sub("", word):
        return "ו/י added or removed at the start of the word"
    return "letters changed beyond adding/removing ו/י"


class VocalizedWord(NamedTuple):
    """One vocalized word of a transcript."""
    plain: str      # the word without nikkud, edge punctuation trimmed
    vocalized: str  # the word as written, edge punctuation trimmed
    line: int       # 1-based line
    context: str    # up to CONTEXT_WORDS words on each side (same line), the word in [brackets]


def _marked_context(words: list[str], index: int) -> str:
    """words[index] in [brackets] (edge punctuation outside) with CONTEXT_WORDS words on each side."""
    word = words[index]
    core = word.strip(_EDGE_PUNCT)
    lead = word[:len(word) - len(word.lstrip(_EDGE_PUNCT))]
    trail = word[len(lead) + len(core):]
    start, end = max(0, index - CONTEXT_WORDS), index + 1 + CONTEXT_WORDS
    parts = words[start:index] + [f"{lead}[{core}]{trail}"] + words[index + 1:end]
    return ("..." if start > 0 else "") + " ".join(parts) + ("..." if end < len(words) else "")


def vocalized_words(text: str) -> list[VocalizedWord]:
    """Every word of `text` that carries nikkud, in order."""
    found = []
    for line_no, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        words = line.split()
        for i, word in enumerate(words):
            core = word.strip(_EDGE_PUNCT)
            if has_nikkud(core):
                found.append(VocalizedWord(strip_nikkud(core), core, line_no, _marked_context(words, i)))
    return found


def _span_context(text: str, start: int, end: int) -> tuple[int, str]:
    """(1-based line, context) of text[start:end]: CONTEXT_WORDS words on each side, the span in [brackets]."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end < 0 else line_end
    before = text[line_start:start].split()
    after = text[end:line_end].split()
    context = (("..." if len(before) > CONTEXT_WORDS else "")
               + " ".join(before[-CONTEXT_WORDS:] + [f"[{text[start:end]}]"] + after[:CONTEXT_WORDS])
               + ("..." if len(after) > CONTEXT_WORDS else ""))
    return text.count("\n", 0, start) + 1, context


WORDS_HEADER = "varies\tplain\tvocalized\tcount\tfirst_line\tcontext"
OVERRIDES_TITLE = "dictionary overrides (the model's nikkud was replaced by the dictionary's)"
OVERRIDES_HEADER = "model\tdictionary\tline\tcontext"


def format_words(words: list[VocalizedWord]) -> str:
    """
    The vocalized-words section: a WORDS_HEADER line, then one tab-separated line per distinct
    (plain, vocalized) pair with its count, first line and the context of its first occurrence.
    varies is "yes" when the same plain word is vocalized more than one way; those words come
    first (each variant on its own line, kept together), then the rest by count.
    """
    pairs: dict[tuple[str, str], list] = {}  # (plain, vocalized) -> [count, first VocalizedWord]
    for word in words:
        pairs.setdefault((word.plain, word.vocalized), [0, word])[0] += 1
    variants = Counter(plain for plain, _ in pairs)
    totals = Counter(word.plain for word in words)
    first_lines = {}
    for word in words:
        first_lines.setdefault(word.plain, word.line)

    def order(item):
        (plain, _), (count, first) = item
        return (variants[plain] == 1, -totals[plain], first_lines[plain], -count, first.line)

    rows = [WORDS_HEADER]
    for (plain, vocalized), (count, first) in sorted(pairs.items(), key=order):
        varies = "yes" if variants[plain] > 1 else "no"
        rows.append(f"{varies}\t{plain}\t{vocalized}\t{count}\t{first.line}\t{first.context}")
    return "\n".join(rows) + "\n"


def format_review(text: str, overrides: list) -> str:
    """
    The review file of a final transcript: the vocalized-words section (format_words), a blank
    line, then the overrides section: an OVERRIDES_TITLE line, an OVERRIDES_HEADER line and one
    tab-separated line per apply_nikkud.NikkudOverride in text order (its line and context in
    `text`, the dictionary's form in [brackets]).
    """
    rows = [OVERRIDES_TITLE, OVERRIDES_HEADER]
    for override in overrides:
        line, context = _span_context(text, override.start, override.start + len(override.dictionary))
        rows.append(f"{override.model}\t{override.dictionary}\t{line}\t{context}")
    return format_words(vocalized_words(text)) + "\n" + "\n".join(rows) + "\n"
