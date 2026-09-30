"""
Nikkud / pronunciation fixes for TTS.

- RABBINIC_NIKKUD_DICT: fixed mappings for rabbinic abbreviations and terms.
- additional_nikkud_corrections: list of (plain, nikkuded) pairs that are expanded
  with common Hebrew prefixes and surrounding space/punctuation into the mapper.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Boundary + prefix expansion helpers
# ---------------------------------------------------------------------------

DEFAULT_PREFIXES: tuple[str, ...] = (
    "",  # bare word
    "ב",
    "מ",
    "כ",
    "ו",
    "ה",
    "מה",
    "ל",
    "וב",
    "וה"
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

    # === Letters ===
    " הא ": " הֵא ",
    " פא ": " פֵּא ",
    " צדי ": " צָדִי ",
}


def build_full_nikkud_map() -> dict[str, str]:
    """Combine fixed rabbinic dict with expanded additional_nikkud_corrections."""
    mapping = dict(RABBINIC_NIKKUD_DICT)
    mapping.update(build_additional_nikkud_map())
    return mapping


def apply_nikkud_to_abbreviations(text: str) -> str:
    """
    Scans the text and replaces Rabbinic acronyms / additional word corrections
    with their Nikkud counterparts.

    Keys are sorted by length descending to prevent partial word replacements.
    """
    mapping = build_full_nikkud_map()
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)

    for key in sorted_keys:
        text = text.replace(key, mapping[key])

    return text


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
