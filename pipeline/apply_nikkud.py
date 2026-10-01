"""
Nikkud / pronunciation fixes for TTS.

- RABBINIC_NIKKUD_DICT: fixed mappings for rabbinic abbreviations and terms.
  These may carry a Hebrew prefix (הרמב"ם, ברמב"ם, מהרמב"ם ...), which is preserved.
- NO_PREFIX_NIKKUD_DICT: standalone words (the letter names הא / פא / צדי) that are
  only replaced as a whole word, never with a prefix.
- additional_nikkud_corrections: list of (plain, nikkuded) pairs; prefixes allowed.

Matching is done with ONE compiled regex (see ``_build_matcher``) using word-boundary
lookarounds, so replacements work at the start/end of the text or a line, next to
punctuation, brackets, quotes and hyphens, but never inside a longer word.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Boundary + prefix expansion helpers
#
# Legacy string-expansion helpers (kept importable). apply_nikkud_to_abbreviations no
# longer uses them: it matches with a boundary-aware regex instead (see below).
# ---------------------------------------------------------------------------

# Single source of truth for the Hebrew prefixes that may precede a replaced word.
# The prefix is kept as-is in the output (prefix + nikkuded replacement).
# The empty string stands for "no prefix" (used by the legacy expansion helpers below;
# the regex matcher treats the prefix as optional and skips it).
DEFAULT_PREFIXES: tuple[str, ...] = (
    "",  # bare word
    # single letters
    "ו", "ה", "ב", "כ", "ל", "מ", "ש",
    # two letters
    "וב", "וה", "וכ", "ול", "ומ", "וש",
    "מה", "מש",
    "שה", "שב", "שכ", "של", "שמ",
    "לה", "בה", "כה", "כש",
    # three+ letters
    "ומה", "ומש", "ושה", "ושב", "ושל", "ושמ",
    "ולה", "ובה", "וכה", "וכש", "כשה", "ולכ", "ולכש", "וכשה",
)

DEFAULT_TRAILING_PUNCT: tuple[str, ...] = (
    "",  # no trailing punctuation
    ".",
    ",",
    "!",
    "?",
    ":",
    ";",
)


def boundary_variants(
    word: str,
    trailing_punct: Sequence[str] | None = None,
) -> list[str]:
    """
    Expand a token into common in-text forms.

    For each trailing punctuation mark, produce a form with a leading space and
    that mark after the word. When there is no trailing punctuation, produce
    a form with a space on both sides.

    Examples for word="שולחן":
      " שולחן ", " שולחן.", " שולחן,", " שולחן!", " שולחן?", ...
    """
    if trailing_punct is None:
        trailing_punct = DEFAULT_TRAILING_PUNCT

    variants: list[str] = []
    seen: set[str] = set()

    def _add(form: str) -> None:
        if form not in seen:
            seen.add(form)
            variants.append(form)

    for punct in trailing_punct:
        if punct == "":
            _add(f" {word} ")
        else:
            _add(f" {word}{punct}")

    return variants


def prefixed_variants(
    word: str,
    prefixes: Sequence[str] | None = None,
    trailing_punct: Sequence[str] | None = None,
) -> list[str]:
    """
    For each prefix (e.g. ב, מ, כ, ו, ה, מה, ל, וב), expand ``prefix + word`` with
    all boundary variants from :func:`boundary_variants`.

    The empty prefix is included by default so the bare word is covered as well.
    """
    if prefixes is None:
        prefixes = DEFAULT_PREFIXES

    variants: list[str] = []
    seen: set[str] = set()
    for prefix in prefixes:
        for form in boundary_variants(prefix + word, trailing_punct=trailing_punct):
            if form not in seen:
                seen.add(form)
                variants.append(form)
    return variants


def expand_correction_pair(
    plain: str,
    nikkuded: str,
    prefixes: Sequence[str] | None = None,
    trailing_punct: Sequence[str] | None = None,
) -> dict[str, str]:
    """
    Expand a single (plain, nikkuded) pair into aligned boundary + prefix forms.

    Example::

        " בשולחן." → " בשׁוּלְחָן."
    """
    plain_forms = prefixed_variants(plain, prefixes=prefixes, trailing_punct=trailing_punct)
    nikkuded_forms = prefixed_variants(nikkuded, prefixes=prefixes, trailing_punct=trailing_punct)
    if len(plain_forms) != len(nikkuded_forms):
        raise ValueError(
            f"Variant count mismatch for pair ({plain!r}, {nikkuded!r}): "
            f"{len(plain_forms)} vs {len(nikkuded_forms)}"
        )
    return dict(zip(plain_forms, nikkuded_forms))


# ---------------------------------------------------------------------------
# Additional word-level corrections (expanded with prefixes + punctuation)
# ---------------------------------------------------------------------------

# Add (plain_or_wrong_form, correctly_nikkuded_form) pairs here.
# Each pair is expanded with prefixes + boundary punctuation into the mapper.
# Example:
#   ("שולחן", "שׁוּלְחָן"),
additional_nikkud_corrections: list[tuple[str, str]] = [
    ("שפתי", "שִׂפְתֵי"),
    ("נבילה", "נְבֵילָה"),
] 


def build_additional_nikkud_map(
    corrections: Iterable[tuple[str, str]] | None = None,
    prefixes: Sequence[str] | None = None,
    trailing_punct: Sequence[str] | None = None,
) -> dict[str, str]:
    """Build the expanded map from ``additional_nikkud_corrections`` (or a custom list)."""
    if corrections is None:
        corrections = additional_nikkud_corrections

    mapping: dict[str, str] = {}
    for plain, nikkuded in corrections:
        mapping.update(
            expand_correction_pair(
                plain,
                nikkuded,
                prefixes=prefixes,
                trailing_punct=trailing_punct,
            )
        )
    return mapping


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


def _canon(key: str) -> str:
    """Normalizes gershayim/geresh to ASCII " and ' so both forms hit the same entry."""
    return key.translate(_CANON_QUOTES)


def _key_to_regex(key: str) -> str:
    """Escapes ``key`` for a regex, letting " match ״ as well and ' match ׳."""
    out = []
    for ch in _canon(key):
        if ch == '"':
            out.append('["\u05F4]')
        elif ch == "'":
            out.append("['\u05F3]")
        else:
            out.append(re.escape(ch))
    return "".join(out)


def _alternation(keys: Iterable[str]) -> str:
    """Longest-first alternation, so a longer key always wins over one of its prefixes."""
    ordered = sorted(set(keys), key=lambda k: (-len(k), k))
    return "|".join(_key_to_regex(k) for k in ordered)


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
    """
    replacements: dict[str, str] = {}
    for plain, nikkuded in standalone:
        replacements[_canon(plain)] = nikkuded
    for plain, nikkuded in prefixable:  # prefixable entries win on a clash
        replacements[_canon(plain)] = nikkuded
    if not replacements:
        return lambda text: text

    prefix_keys = [_canon(p) for p, _ in prefixable]
    prefix_list = sorted({p for p in prefixes if p}, key=lambda p: (-len(p), p))

    alternatives = [f"(?P<bare>{_alternation(replacements)})"]
    if prefix_keys and prefix_list:
        pre = "|".join(re.escape(p) for p in prefix_list)
        alternatives.append(f"(?P<pre>{pre})(?P<key>{_alternation(prefix_keys)})")
    pattern = re.compile(_BEFORE + "(?:" + "|".join(alternatives) + ")" + _AFTER)

    def _sub(match: re.Match) -> str:
        bare = match.group("bare")
        if bare is not None:
            return replacements[_canon(bare)]
        return match.group("pre") + replacements[_canon(match.group("key"))]

    return lambda text: pattern.sub(_sub, text)


def build_full_nikkud_map() -> dict[str, str]:
    """
    Combine all entries into one ``plain -> nikkuded`` map: RABBINIC_NIKKUD_DICT,
    NO_PREFIX_NIKKUD_DICT and additional_nikkud_corrections.

    This is the *unexpanded* map (no prefixes / spacing variants): prefixes and word
    boundaries are handled by the regex matcher in :func:`apply_nikkud_to_abbreviations`.
    """
    mapping = dict(RABBINIC_NIKKUD_DICT)
    mapping.update(NO_PREFIX_NIKKUD_DICT)
    mapping.update(additional_nikkud_corrections)
    return mapping


def apply_nikkud_to_abbreviations(text: str) -> str:
    """
    Replaces Rabbinic acronyms / additional word corrections with their nikkud forms.

    Each entry matches as a whole word (see the boundary rule above), optionally preceded
    by one of DEFAULT_PREFIXES, which is preserved. Letter entries (NO_PREFIX_NIKKUD_DICT)
    match only as standalone words. Already-nikkuded text is not matched again, so the
    function is idempotent.
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
