"""
Nikkud / pronunciation fixes for TTS.

- RABBINIC_NIKKUD_DICT: fixed mappings for rabbinic abbreviations and terms.
  These may carry a Hebrew prefix (הרמב"ם, ברמב"ם, להרמב"ם, מהרמב"ם ...), which is preserved.
- NO_PREFIX_NIKKUD_DICT: the letter names הא / פא / צדי. Replaced only as a whole word,
  never with a prefix, and only in a letter-name context (see LETTER-NAME CONTEXT below).
- additional_nikkud_corrections: list of (plain, nikkuded) pairs; the same prefixes are allowed.

All prefixed entries share ONE prefix set, DEFAULT_PREFIXES (generated from a single rule).

Matching is done with ONE compiled regex (see ``_build_matcher``) using word-boundary
lookarounds, so replacements work at the start/end of the text or a line, next to
punctuation, brackets, quotes and hyphens, but never inside a longer word.

Multi-word keys (e.g. "אבן העזר"): a space inside a key matches any run of spaces, tabs and
NBSP (``_BLANK``+), never a line break (\\n, \\r); the output is the dict value as written.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Iterable

# ---------------------------------------------------------------------------
# Hebrew prefixes (shared by every prefixable entry)
# ---------------------------------------------------------------------------

# Single source of truth for the Hebrew prefixes that may precede a replaced word.
# The prefix is kept as-is in the output (prefix + nikkuded replacement).
# The empty string stands for "no prefix"; the regex matcher treats the prefix as
# optional and filters "" out when building the alternation.
#
# The set is GENERATED from this rule (see _generate_prefixes):
#
#     prefix = [ו] [core] [ה]
#
#   core  = ב | כ | ל | מ                      single prepositions
#         | ש | כש                             "that" / "when"
#         | ש or כש + a preposition            שב שכ של שמ  כשב כשל כשמ
#                                              (כש already contains כ, so no כשכ)
#         | מש                                 מ + ש
#   ה     = the definite article, written out. It is accepted after nothing (ה), after ו
#           (וה), and after any core, including ב / כ / ל (בה כה לה שלה ובה ולה וכה ...):
#           names and abbreviations keep their ה (להרמב"ם, כהרשב"א, בהרא"ש).
#   Everything is optional, but a non-empty prefix has at least one piece.
#   Result: 60 entries (the empty prefix plus 59 non-empty ones).
_PREPOSITIONS = "בכלמ"
_SHIN_FORMS = ("ש", "כש")


def _generate_prefixes() -> tuple[str, ...]:
    """Generate the prefix set from the rule above (sorted by length, then alphabetically)."""
    cores = ["", *_PREPOSITIONS, *_SHIN_FORMS, "מש"]
    for shin in _SHIN_FORMS:
        for prep in _PREPOSITIONS:
            if prep not in shin:  # כש already contains כ
                cores.append(shin + prep)

    prefixes = set()
    for vav in ("", "ו"):
        for core in cores:
            prefixes.add(vav + core)
            prefixes.add(vav + core + "ה")
    prefixes.discard("")
    return ("",) + tuple(sorted(prefixes, key=lambda p: (len(p), p)))


DEFAULT_PREFIXES: tuple[str, ...] = _generate_prefixes()

# ---------------------------------------------------------------------------
# Additional word-level corrections (same prefix set + regex boundaries as abbreviations)
# ---------------------------------------------------------------------------

# Add (plain_or_wrong_form, correctly_nikkuded_form) pairs here.
# Each pair is matched by the regex matcher like RABBINIC_NIKKUD_DICT entries.
# Example:
#   ("שולחן", "שׁוּלְחָן"),
additional_nikkud_corrections: list[tuple[str, str]] = [
    ("שפתי כהן", "שִׂפְתֵי כהן"),
    ("נבילה", "נְבֵילָה"),
    ("בעין", "בָּעַיִן"),
    ("בית יוסף", "בֵּית יוֹסֵף"),
    ("פרווה", "פַּרְוֶה"),
    ("קדירה", "קְדֵרָה"),
    ("קדרה", "קְדֵרָה"),
    ('סאין', 'סְאִין'),
    ('כלאים', 'כִּלְאַיִם'),
    ('כותח', 'כּוּתָח'),
    ('דמיה', 'דָּמֶיהָ'),

]


# ---------------------------------------------------------------------------
# Fixed rabbinic abbreviation dictionary
# ---------------------------------------------------------------------------

# A comprehensive dictionary of Halachic abbreviations and their precise TTS equivalents
RABBINIC_NIKKUD_DICT = {
    # === Rishonim (ראשונים) ===
    'רש"י': 'רַשִׁי',
    'רמב"ם': 'רַמְבַּם',
    'רמב"ן': 'רַמְבַּן',
    'רשב"א': 'רַשְׁבָּא',
    'ריטב"א': 'רִיטְבָּא',
    'רשב"ם': 'רַשְׁבַּם',
    'ראב"ד': 'רַאֲבַד',  # Fixed: בלי דגש בב'
    'רא"ש': 'רֹאשׁ',
    'רי"ף': 'רִיף',
    'ר"ת': 'רַבֵּינוּ תַּם',  # Fixed: נפתח למילה מלאה
    'ר"ן': 'רַן',
    'רז"ה': 'בַּעַל הַמָּאוֹר',  # Fixed: הוחלף לשם הספר
    'ריב"ש': 'רִיבָשׁ',  # Fixed: בלי דגש בב'
    'תשב"ץ': 'תַּשְׁבֵּץ',
    'רד"ק': 'רַדַק',
    'רלב"ג': 'רַלְבַּג',
    'רמ"ה': 'יַד רָמָה',  # Fixed: הוחלף לשם הספר
    'סמ"ג': 'סְמַג',
    'סמ"ק': 'סְמַק',
    'רמ"ך': 'רָמָךְ',
    'הר"ר פרץ': 'הרב רַבֵּנוּ פֶּרֶץ'',
    'רבינו פרץ': 'רַבֵּנוּ פֶּרֶץ'

    # === Acharonim (אחרונים) ===
    'רמ"א': 'רַמָא',
    'ש"ך': 'שַׁךְ',
    'ט"ז': 'טַז',
    'סמ"ע': 'סְמַע',
    'ב"ח': 'בַּח',
    'מג"א': 'מָגֵן אַבְרָהָם',  # Fixed: נפתח
    'מהרש"א': 'מַהַרְשָׁא',
    'מהרש"ל': 'מַהַרְשַׁל',
    'מהר"ם': 'מַהֲרַם',
    'מהרי"ל': 'מַהֲרִיל',
    'מהרי"ט': 'מַהֲרִיט',
    'רדב"ז': 'רַדְבַּז',
    'חיד"א': 'חִידָא',
    'חת"ס': 'חֲתַם סוֹפֵר',  # Fixed: נפתח
    'גר"א': 'גְּרָא',
    'נצי"ב': 'נָצִיב',
    'חזו"א': 'חֲזוֹן אִישׁ',  # Fixed: נפתח
    'פרמ"ג': 'פְּרִי מְגָדִים',  # Fixed: נפתח
    'נוב"י': 'נוֹדָע בִּיהוּדָה',  # Fixed: נפתח
    'כה"ח': 'כַּף הַחַיִּים',  # Fixed: נפתח
    'משנ"ב': 'מִשְׁנָה בְּרוּרָה',  # Fixed: נפתח
    'ערוה"ש': 'עֲרוּךְ הַשֻּׁלְחָן',  # Fixed: נפתח

    # === Major Halachic Works & Terms ===
    'שו"ע': 'שׁוּלְחָן עָרוּךְ',  # Fixed: נפתח
    'ב"י': 'בֵּית יוֹסֵף',  # Fixed: נפתח
    'יו"ד': 'יוֹרֶה דֵּעָה',  # Fixed: נפתח
    'או"ח': 'אוֹרַח חַיִּים',  # Fixed: נפתח
    'אבן העזר': 'אֶבֶן הָעֵזֶר',
    'אה"ע': 'אֶבֶן הָעֵזֶר',  # Fixed: נפתח
    'חו"מ': 'חוֹשֶׁן מִשְׁפָּט',  # Fixed: נפתח
    'חנ"ן': 'חֲנָן',
    'נ"ט': 'נַט',  # Fixed: הושאר כקיצור עם ניקוד קריא
    'ע"פ': 'עַל פִּי',
    'אע"פ': 'אַף עַל פִּי',
    'שו"ת': 'שׁוּ"ת',
}

# Standalone words: replaced only as a whole word, never with a prefix
# (a bare הא / פא / צדי is already risky; ב+הא etc. is meaningless).
NO_PREFIX_NIKKUD_DICT = {
    'הא': 'הֵא',
    'פא': 'פֵּא',
    'צדי': 'צָדִי',
}


# ---------------------------------------------------------------------------
# Regex matcher
# ---------------------------------------------------------------------------

# "Word characters" for boundary purposes: Hebrew letters (U+05D0-05EA) and Hebrew
# points / nikkud / cantillation marks (U+0591-05C7), EXCEPT the punctuation-like
# maqaf (U+05BE), paseq (U+05C0) and sof pasuq (U+05C3), which act as separators
# (so שפתי־כהן behaves like שפתי-כהן).
_WORD_CHARS = "\u05D0-\u05EA\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4-\u05C7"

# Quote-like characters that occur INSIDE abbreviations (רמב"ם, ג'): ASCII ", Hebrew
# gershayim ״ (U+05F4), ASCII apostrophe ' and geresh ׳ (U+05F3).
_QUOTE_CHARS = "\"\u05F4'\u05F3"

# Spaces inside a key are not part of the boundary rule below: they match a run of
# blanks (space / tab / NBSP, see _BLANK), never a line break. Matched text is normalized
# with _norm (quotes canonicalized, blank runs collapsed to one space) before dict lookup.
#
# Boundary rule (applied to the whole match, prefix included):
#   BEFORE: the match must not be preceded by a word char, and must not be preceded
#           by <word char><quote> (a quote glued to the end of a previous letter is
#           inside a word, e.g. ה"שפתי or the ר"ן inside הגר"ן).
#           A quote preceded by a space / bracket / start of text is just a wrapping
#           quote (e.g. "שפתי" or 'שפתי') and does NOT block the match.
#   AFTER:  mirrored: not followed by a word char, and not followed by <quote><word char>.
#           (שפתי" at the end of a quotation matches; שפתי"ם does not.)
_BEFORE = f"(?<![{_WORD_CHARS}])(?<![{_WORD_CHARS}][{_QUOTE_CHARS}])"
_AFTER = f"(?![{_WORD_CHARS}])(?![{_QUOTE_CHARS}][{_WORD_CHARS}])"

_CANON_QUOTES = str.maketrans({"\u05F4": '"', "\u05F3": "'"})

# Blank space that does NOT cross a line: space, tab, NBSP. Never \n or \r.
# A space inside a dict key (אבן העזר, בית יוסף) matches a run of one or more of these.
_BLANK = "[ \t\u00A0]"
_BLANK_RUN = re.compile(f"{_BLANK}+")


def _canon(key: str) -> str:
    """Normalizes gershayim/geresh to ASCII " and ' so both forms hit the same entry."""
    return key.translate(_CANON_QUOTES)


def _norm(text: str) -> str:
    """Canonical form used for dict lookups: canon quotes + each run of blanks -> one ASCII space."""
    return _BLANK_RUN.sub(" ", _canon(text))


def _key_to_regex(key: str) -> str:
    """
    Escapes ``key`` for a regex, letting " match ״ as well and ' match ׳.

    A space in the key matches a run of one or more blanks (space / tab / NBSP), never a
    line break, so "אבן העזר" also matches "אבן  העזר" and "אבן\u00A0העזר" but not "אבן\nהעזר".
    """
    out = []
    for ch in _norm(key):
        if ch == '"':
            out.append('["\u05F4]')
        elif ch == "'":
            out.append("['\u05F3]")
        elif ch == " ":
            out.append(_BLANK + "+")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _alternation(keys: Iterable[str]) -> str:
    """Longest-first alternation, so a longer key always wins over one of its prefixes."""
    ordered = sorted(set(keys), key=lambda k: (-len(k), k))
    return "|".join(_key_to_regex(k) for k in ordered)


# ---------------------------------------------------------------------------
# LETTER-NAME CONTEXT (NO_PREFIX_NIKKUD_DICT entries: הא / פא / צדי)
#
# These are ordinary Hebrew/Aramaic words too (הא דתניא, הא קמ"ל), so they are replaced
# only when the text clearly means a letter name used as a number (סימן צדי הא, אות פא):
#   1. preceded, on the SAME line, by a context word (LETTER_NAME_CONTEXT_WORDS) - optionally
#      carrying a prefix (בסימן, לסעיף, ובאות ...) - with only whitespace, quotes, a
#      colon, a dash or a bracket between them (not . , ; ? ! - those end the phrase), or
#   2. directly adjacent (separated only by blank space) to another letter-name word
#      (LETTER_NAMES), before or after: "צדי הא", "פא זין", "הא ריש".
# A trailing geresh (הא' = "the first", an ordinal) is never a boundary for these entries,
# unless the word is wrapped in quotes ('הא').
# Anything else - "הא דתניא", a sentence-initial "הא" - is left unchanged.
# Known residual: "הא תו ..." (Aramaic "הא תו") counts as adjacent to the letter name תו.
# ---------------------------------------------------------------------------

LETTER_NAME_CONTEXT_WORDS: tuple[str, ...] = ("סימן", "סעיף", 'ס"ק', "דף", "פרק", "אות", "סי'")

LETTER_NAMES: tuple[str, ...] = (
    "צדי", "צדיק", "טית", "זין", "וו", "דלת", "גימל", "בית", "אלף", "חית", "יוד", "כף",
    "למד", "מם", "נון", "סמך", "עין", "פא", "קוף", "ריש", "שין", "תו", "הא",
)

# allowed between a context word and the letter name (no sentence punctuation)
_CONTEXT_GAP = (
    "[ \t\u00A0\"\u05F4'\u05F3:\\-\u05BE\u2013\u2014()\\[\\]]{0,4}"
)
_WORD_CLASS = f"[{_WORD_CHARS}]"
_LOOKBACK = 48  # chars of the current line examined before a match (keeps matching O(n))


@lru_cache(maxsize=None)
def _letter_context_regexes(prefixes: tuple[str, ...]):
    """Compiled (context_before, name_before, name_after) regexes for the context rule."""
    ctx = "|".join(_key_to_regex(w) for w in sorted(LETTER_NAME_CONTEXT_WORDS, key=lambda w: (-len(w), w)))
    names = "|".join(_key_to_regex(n) for n in sorted(LETTER_NAMES, key=lambda w: (-len(w), w)))
    pre = "|".join(re.escape(p) for p in sorted({p for p in prefixes if p}, key=lambda p: (-len(p), p)))
    context_before = re.compile(
        f"(?<![{_WORD_CHARS}])(?:{pre})?(?:{ctx}){_CONTEXT_GAP}$"
    )
    # a neighbour that is half of a hyphenated compound (בית-דין, זין-X) is not a letter name
    name_before = re.compile(f"(?<![{_WORD_CHARS}\\-\u05BE])(?:{names})[{_QUOTE_CHARS}]?{_BLANK}+$")
    name_after = re.compile(f"^[{_QUOTE_CHARS}]?{_BLANK}+(?:{names}){_AFTER}(?![\\-\u05BE]{_WORD_CLASS})")
    return context_before, name_before, name_after


def _has_letter_name_context(text: str, start: int, end: int, prefixes: tuple[str, ...]) -> bool:
    """True if text[start:end] (a הא / פא / צדי match) is in a letter-name context."""
    # a trailing geresh makes it an ordinal (הא'), unless the word is wrapped in quotes ('הא')
    if text[end:end + 1] in ("'", "\u05F3") and text[start - 1:start] not in ("'", "\u05F3", '"', "\u05F4"):
        return False
    context_before, name_before, name_after = _letter_context_regexes(prefixes)
    line_start = text.rfind("\n", 0, start) + 1
    line_start = max(line_start, text.rfind("\r", 0, start) + 1, start - _LOOKBACK)
    before = text[line_start:start]
    if context_before.search(before) or name_before.search(before):
        return True
    after_end = text.find("\n", end)
    after = text[end:end + _LOOKBACK if after_end < 0 else min(after_end, end + _LOOKBACK)]
    return bool(name_after.search(after))


@lru_cache(maxsize=None)
def _build_matcher(
    prefixable: tuple[tuple[str, str], ...],
    standalone: tuple[tuple[str, str], ...],
    prefixes: tuple[str, ...],
) -> Callable[[str], str]:
    """
    Compiles ONE combined regex (cached) and returns a ``text -> text`` function.

    At each position the regex first tries a bare key (any entry, longest first), and
    only then ``prefix + key`` (prefixable entries only, longest prefix first, with
    backtracking to shorter ones). So מהר"ם (an entry) is never read as מה+ר"ם, and
    bare ב"ח is never double-prefixed, while הב"ח is ה + ב"ח.
    ``standalone`` entries (letter names) match only bare and only in a letter-name context.
    """
    replacements: dict[str, str] = {}
    for plain, nikkuded in standalone:
        replacements[_norm(plain)] = nikkuded
    for plain, nikkuded in prefixable:  # prefixable entries win on a clash
        replacements[_norm(plain)] = nikkuded
    if not replacements:
        return lambda text: text
    prefix_keys = [_norm(p) for p, _ in prefixable]
    context_only = {_norm(p) for p, _ in standalone} - set(prefix_keys)

    prefix_list = sorted({p for p in prefixes if p}, key=lambda p: (-len(p), p))
    alternatives = [f"(?P<bare>{_alternation(replacements)})"]
    if prefix_keys and prefix_list:
        pre = "|".join(re.escape(p) for p in prefix_list)
        alternatives.append(f"(?P<pre>{pre})(?P<key>{_alternation(prefix_keys)})")
    pattern = re.compile(_BEFORE + "(?:" + "|".join(alternatives) + ")" + _AFTER)

    def _sub(match: re.Match) -> str:
        bare = match.group("bare")
        if bare is not None:
            key = _norm(bare)
            if key in context_only and not _has_letter_name_context(
                match.string, match.start(), match.end(), prefixes
            ):
                return match.group(0)
            return replacements[key]
        return match.group("pre") + replacements[_norm(match.group("key"))]

    return lambda text: pattern.sub(_sub, text)


def apply_nikkud_to_abbreviations(text: str) -> str:
    """
    Replaces Rabbinic acronyms / additional word corrections with their nikkud forms.

    Each entry matches as a whole word (see the boundary rule above), optionally preceded
    by one of DEFAULT_PREFIXES, which is preserved. Letter entries (NO_PREFIX_NIKKUD_DICT)
    match only as standalone words and only in a letter-name context. Already-nikkuded text
    is not matched again, so the function is idempotent.
    """
    prefixable = tuple(
        list(RABBINIC_NIKKUD_DICT.items()) + list(additional_nikkud_corrections)
    )
    matcher = _build_matcher(
        prefixable,
        tuple(NO_PREFIX_NIKKUD_DICT.items()),
        tuple(DEFAULT_PREFIXES),
    )
    return matcher(text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply Hebrew nikkud / pronunciation fixes to a text file."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input Hebrew text file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Path to save the result (optional; defaults to appending '_nikkud')",
    )
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        content = f.read()

    result = apply_nikkud_to_abbreviations(content)

    output_path = args.output_file or args.input_file.replace(".txt", "_nikkud.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Nikkud applied and saved to {output_path}")
