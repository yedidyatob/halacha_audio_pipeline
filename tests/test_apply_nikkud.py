"""Tests for pipeline/apply_nikkud.py (regex word-boundary matching with Hebrew prefixes)."""
import pytest

from pipeline import apply_nikkud as an
from pipeline.apply_nikkud import (
    DEFAULT_PREFIXES,
    NO_PREFIX_NIKKUD_DICT,
    RABBINIC_NIKKUD_DICT,
    additional_nikkud_corrections,
    apply_nikkud_to_abbreviations as apply,
    build_full_nikkud_map,
)

# Expected values come from the module's own tables (avoids Unicode mark-order typos).
SHAFTEI = dict(additional_nikkud_corrections)["שפתי"]
NEVILA = dict(additional_nikkud_corrections)["נבילה"]
RAMBAM = RABBINIC_NIKKUD_DICT['רמב"ם']


# ---------------------------------------------------------------------------
# Public API still importable / working
# ---------------------------------------------------------------------------

def test_public_api_importable():
    assert callable(an.expand_correction_pair)
    assert callable(an.build_additional_nikkud_map)
    assert an.expand_correction_pair("שפתי", SHAFTEI)[" שפתי "] == f" {SHAFTEI} "
    full = build_full_nikkud_map()
    assert full['רמב"ם'] == RAMBAM and full["הא"] == NO_PREFIX_NIKKUD_DICT["הא"] and full["שפתי"] == SHAFTEI
    assert additional_nikkud_corrections and RABBINIC_NIKKUD_DICT and NO_PREFIX_NIKKUD_DICT


def test_prefix_list_is_clean():
    assert len(DEFAULT_PREFIXES) == len(set(DEFAULT_PREFIXES))
    for p in DEFAULT_PREFIXES:
        assert all("\u05D0" <= c <= "\u05EA" for c in p)
    for expected in ("ו", "ה", "ב", "כ", "ל", "מ", "ש", "וב", "וה", "מה", "שה", "ול", "וכש", "כש"):
        assert expected in DEFAULT_PREFIXES


# ---------------------------------------------------------------------------
# Boundaries for an additional correction word (שפתי)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("זה שפתי כהן", f"זה {SHAFTEI} כהן"),                 # mid-sentence
    ("שפתי כהן אומר", f"{SHAFTEI} כהן אומר"),             # start of text
    ("שורה ראשונה\nשפתי כהן", f"שורה ראשונה\n{SHAFTEI} כהן"),  # start of line
    ("אמר שפתי\nוהלאה", f"אמר {SHAFTEI}\nוהלאה"),         # end of line
    ("כך אמר שפתי", f"כך אמר {SHAFTEI}"),                 # end of text
    ("שפתי.", f"{SHAFTEI}."),
    ("שפתי,", f"{SHAFTEI},"),
    ("שפתי;", f"{SHAFTEI};"),
    ("שפתי:", f"{SHAFTEI}:"),
    ("שפתי!", f"{SHAFTEI}!"),
    ("שפתי?", f"{SHAFTEI}?"),
    ("(שפתי)", f"({SHAFTEI})"),
    ("[שפתי]", f"[{SHAFTEI}]"),
    ('"שפתי"', f'"{SHAFTEI}"'),
    ("'שפתי'", f"'{SHAFTEI}'"),
    ('אמר "שפתי כהן" שם', f'אמר "{SHAFTEI} כהן" שם'),
    ("שפתי-כהן", f"{SHAFTEI}-כהן"),
    ("כהן-שפתי", f"כהן-{SHAFTEI}"),
    ("שפתי־כהן", f"{SHAFTEI}־כהן"),                       # maqaf is a separator
    ("שפתי שפתי", f"{SHAFTEI} {SHAFTEI}"),
    ("שפתי\tשפתי\r\nשפתי", f"{SHAFTEI}\t{SHAFTEI}\r\n{SHAFTEI}"),
])
def test_boundaries(text, expected):
    assert apply(text) == expected


@pytest.mark.parametrize("prefix", ["ב", "ו", "ל", "מ", "ה", "כ", "ש", "וב", "וה", "ול", "מה", "שה", "וכש", "כש", "ולכ"])
def test_prefixes_are_preserved(prefix):
    assert apply(f"אמר {prefix}שפתי כהן") == f"אמר {prefix}{SHAFTEI} כהן"
    assert apply(f"{prefix}שפתי") == f"{prefix}{SHAFTEI}"
    assert apply(f"{prefix}נבילה.") == f"{prefix}{NEVILA}."


def test_specific_prefix_examples():
    assert apply("בשפתי") == "ב" + SHAFTEI
    assert apply("ושפתי") == "ו" + SHAFTEI
    assert apply("ולשפתי") == "ול" + SHAFTEI
    assert apply("ששפתי") == "ש" + SHAFTEI
    assert apply("מהשפתי") == "מה" + SHAFTEI


@pytest.mark.parametrize("text", [
    "הבשפתי",       # not a valid prefix combination
    "שפתיים",       # longer word
    "שפתיו",
    "ישפתי",
    "גשפתי",
    "שפתי\"ם",      # quote glued to the following letter = inside a word
    "ה\"שפתי",      # quote glued to the preceding letter
    "נבילות",
    "קנבילה",
])
def test_not_replaced_inside_longer_word(text):
    assert apply(text) == text


def test_already_nikkuded_words_are_not_touched():
    assert apply(SHAFTEI) == SHAFTEI
    assert apply("בשִׂפְתֵי") == "בשִׂפְתֵי"


# ---------------------------------------------------------------------------
# Abbreviations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ('רמב"ם', RAMBAM),
    ('הרמב"ם', "ה" + RAMBAM),
    ('ברמב"ם', "ב" + RAMBAM),
    ('מהרמב"ם', "מה" + RAMBAM),
    ('והרמב"ם', "וה" + RAMBAM),
    ('כרמב"ם', "כ" + RAMBAM),
    ('ט"ז', "טַז"),
    ('הט"ז', "הטַז"),
    ('ב"ח', "בַּח"),
    ('הב"ח', "הבַּח"),
    ('ובב"ח', "וב" + "בַּח"),
    ('ב"י', "בֵּית יוֹסֵף"),
    ('הב"י', "ה" + "בֵּית יוֹסֵף"),
    ('מג"א', "מָגֵן אַבְרָהָם"),
    ('המג"א', "ה" + "מָגֵן אַבְרָהָם"),
    ('מהר"ם', "מַהֲרַם"),
    ('מהרש"א', "מַהַרְשָׁא"),
    ('ע"פ', "עַל פִּי"),
    ('בע"פ', "ב" + "עַל פִּי"),
    ('אע"פ', "אַף עַל פִּי"),
    ('ואע"פ', "ו" + "אַף עַל פִּי"),
    ('אבן העזר', "אֶבֶן הָעֵזֶר"),
    ('באבן העזר', "ב" + "אֶבֶן הָעֵזֶר"),
    ('שו"ע', "שׁוּלְחָן עָרוּךְ"),
    ('בשו"ע', "ב" + "שׁוּלְחָן עָרוּךְ"),
    ('שו"ת', 'שׁוּ"ת'),
])
def test_abbreviations(text, expected):
    assert apply(text) == expected


def test_abbreviations_in_sentences_and_punctuation():
    assert apply('כתב הרמב"ם, וכן הט"ז.') == f"כתב ה{RAMBAM}, וכן הטַז."
    assert apply('(רמב"ם)') == f"({RAMBAM})"
    assert apply('"רמב"ם"') == f'"{RAMBAM}"'          # wrapping quotes + inner gershayim
    assert apply('רמב"ם\nט"ז') == f"{RAMBAM}\nטַז"
    assert apply('רמב"ם-ט"ז') == f"{RAMBAM}-טַז"
    assert apply('הרמב"ם:') == f"ה{RAMBAM}:"


def test_inner_quote_does_not_split_abbreviation():
    # The quote inside the abbreviation must not be treated as a boundary.
    assert apply('רמב"ם') == RAMBAM
    assert apply('רמב"ם') != 'רמב"ם'
    assert apply('רמב"ם ורמב"ן') == f"{RAMBAM} ו" + "רַמְבַּן"


def test_gershayim_form_is_equivalent_to_ascii_quote():
    assert apply("רמב״ם") == RAMBAM
    assert apply("הרמב״ם") == "ה" + RAMBAM
    assert apply("מהר״ת") == apply('מהר"ת')
    assert apply("אע״פ") == "אַף עַל פִּי"


@pytest.mark.parametrize("text", [
    'הגר"ן',        # ר"ן is preceded by the letter ג -> inside a longer word
    'הגר"ן הגדול',
    'גר"ן',
    'הרמב"ם"ל',     # trailing letter after the closing quote -> still one word
    'ארמב"ם',       # not a valid prefix (א)
    'רמב"מים',
    'שוט"ז',
])
def test_abbreviation_not_matched_inside_longer_word(text):
    assert apply(text) == text


def test_mehar_tav_is_prefix_mem_he_plus_ar_tav():
    """
    מהר"ת = מה (prefix) + ר"ת. With the 'מה' prefix this becomes מה + רַבֵּינוּ תַּם
    ("מהרבינו תם"), which is plausible Hebrew. The old code produced the wrong
    מהרַבֵּינוּ תַּם via mid-word matching of ר"ת, which looked identical; the point of the
    new design is that this now happens only because a legitimate prefix is present.
    """
    assert apply('מהר"ת') == "מה" + "רַבֵּינוּ תַּם"
    assert apply('הר"ת') == "ה" + "רַבֵּינוּ תַּם"
    assert apply('ר"ת') == "רַבֵּינוּ תַּם"
    # ...but not when the letters before are not a valid prefix combination:
    assert apply('זהר"ת') == 'זהר"ת'
    assert apply('אהר"ת') == 'אהר"ת'


def test_entries_starting_with_prefix_letters_are_not_double_prefixed():
    # bare entries win over prefix + shorter entry
    assert apply('מהר"ם') == "מַהֲרַם"            # NOT מ + הר"ם
    assert apply('מהרש"א') == "מַהַרְשָׁא"        # NOT מ + הרש"א
    assert apply('מג"א') == "מָגֵן אַבְרָהָם"
    assert apply('ב"ח') == "בַּח"                 # not ב + "ח
    assert apply('ב"י') == "בֵּית יוֹסֵף"
    assert apply('אע"פ') == "אַף עַל פִּי"        # NOT א + ע"פ
    assert apply('בע"פ') == "ב" + "עַל פִּי"      # ב + ע"פ (בעל פה-style abbreviation context)


# ---------------------------------------------------------------------------
# Letter entries: standalone only, no prefixes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("הא", "הֵא"),
    ("פא", "פֵּא"),
    ("צדי", "צָדִי"),
    ("האות הא.", "האות הֵא."),
    ("אות פא,", "אות פֵּא,"),
    ("אות צדי\nוהלאה", "אות צָדִי\nוהלאה"),
    ("הא\nפא", "הֵא\nפֵּא"),
    ('"צדי"', '"צָדִי"'),
    ("(הא)", "(הֵא)"),
    ("הא-פא", "הֵא-פֵּא"),
])
def test_letter_entries_standalone(text, expected):
    assert apply(text) == expected


@pytest.mark.parametrize("text", [
    "בהא", "והא", "להא", "מהא", "שהא", "הפא", "בפא", "וצדי", "בצדי", "מצדי",
    "האדם", "פאה", "צדיק", "צדיקים", "ופא",
])
def test_letter_entries_not_matched_with_prefix_or_inside_words(text):
    assert apply(text) == text


# ---------------------------------------------------------------------------
# Idempotence, empty, mixed text
# ---------------------------------------------------------------------------

def test_empty_and_plain_text():
    assert apply("") == ""
    assert apply("שלום עולם. אין כאן קיצורים.") == "שלום עולם. אין כאן קיצורים."


def test_idempotent_on_every_entry():
    entries = (
        list(RABBINIC_NIKKUD_DICT) + list(NO_PREFIX_NIKKUD_DICT)
        + [p for p, _ in additional_nikkud_corrections]
    )
    text = "\n".join(f"הנה {key} כאן, ו{key}." for key in entries)
    once = apply(text)
    assert once != text
    assert apply(once) == once


def test_idempotent_mixed_paragraph():
    text = 'כתב הרמב"ם בשפתי כהן, ומהר"ת לא כמו הט"ז.\n"נבילה" היא אות הא ו(צדי) וגם מהרש"ל.\nשו"ת וחו"מ'
    once = apply(text)
    assert apply(once) == once
    assert 'רמב"ם' not in once and "שפתי" not in once


def test_every_dict_key_replaced_standalone_and_with_each_prefix_sample():
    for key, value in RABBINIC_NIKKUD_DICT.items():
        assert apply(f"א {key} ב") == f"א {value} ב", key
        assert apply(f"א ה{key} ב").endswith(" ב"), key
    for key, value in NO_PREFIX_NIKKUD_DICT.items():
        assert apply(f"א {key} ב") == f"א {value} ב", key
        assert apply(f"א ב{key} ב") == f"א ב{key} ב", key


def test_matcher_is_cached():
    an._build_matcher.cache_clear()
    apply("שפתי")
    apply("רמב\"ם")
    info = an._build_matcher.cache_info()
    assert info.misses == 1 and info.hits >= 1


def test_custom_corrections_list_is_picked_up(monkeypatch):
    monkeypatch.setattr(an, "additional_nikkud_corrections", [("שולחן", "שֻׁלְחָן")])
    assert an.apply_nikkud_to_abbreviations("בשולחן") == "בשֻׁלְחָן"
    assert an.apply_nikkud_to_abbreviations("שפתי") == "שפתי"
