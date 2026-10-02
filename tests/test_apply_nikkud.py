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
    assert DEFAULT_PREFIXES[0] == ""  # bare word
    assert len(DEFAULT_PREFIXES) == len(set(DEFAULT_PREFIXES))
    for p in DEFAULT_PREFIXES:
        assert all("\u05D0" <= c <= "\u05EA" for c in p)
    for expected in ("ו", "ה", "ב", "כ", "ל", "מ", "ש", "וב", "וה", "מה", "שה", "ול", "וכש", "כש"):
        assert expected in DEFAULT_PREFIXES


def test_generated_prefix_set_is_exactly_the_rule():
    # prefix = [ו] [core] [ה]; every piece optional, a non-empty prefix has at least one
    cores = {
        "", "ב", "כ", "ל", "מ", "ש", "כש",            # single prepositions, "that", "when"
        "שב", "שכ", "של", "שמ", "כשב", "כשל", "כשמ",   # ש / כש + preposition (כש contains כ: no כשכ)
        "מש",
    }
    expected = {vav + core + he for vav in ("", "ו") for core in cores for he in ("", "ה")} - {""}
    assert set(DEFAULT_PREFIXES) == expected | {""}
    assert DEFAULT_PREFIXES[0] == ""
    assert len(DEFAULT_PREFIXES) == len(set(DEFAULT_PREFIXES)) == 60   # "" + 59 non-empty


def test_prefix_set_is_listed_explicitly():
    assert sorted(DEFAULT_PREFIXES[1:]) == sorted([
        # no ו
        "ב", "כ", "ל", "מ", "ש", "ה", "כש", "שב", "שכ", "של", "שמ", "כשב", "כשל", "כשמ", "מש",
        "בה", "כה", "לה", "מה", "שה", "כשה", "שבה", "שכה", "שלה", "שמה", "כשבה", "כשלה", "כשמה", "משה",
        # with ו
        "ו", "וב", "וכ", "ול", "ומ", "וש", "וה", "וכש", "ושב", "ושכ", "ושל", "ושמ", "וכשב", "וכשל", "וכשמ", "ומש",
        "ובה", "וכה", "ולה", "ומה", "ושה", "וכשה", "ושבה", "ושכה", "ושלה", "ושמה", "וכשבה", "וכשלה", "וכשמה", "ומשה",
    ])


def test_generator_has_no_special_cases_left():
    """One rule, one set, one exported name: no flag, no second set."""
    import inspect
    assert not hasattr(an, "ABBREVIATION_PREFIXES")
    assert not inspect.signature(an._generate_prefixes).parameters
    assert an._generate_prefixes() == DEFAULT_PREFIXES


@pytest.mark.parametrize("invalid", [
    "ולכ", "ולכש", "בל", "כב", "בב", "ההה", "הו", "וו", "שש", "כשכ", "כשש", "בהה", "להל", "כהב",
    "בהב", "ובו", "הב", "המ", "הכ", "שוב", "מכ",
])
def test_invalid_prefixes_not_generated(invalid):
    assert invalid not in DEFAULT_PREFIXES


@pytest.mark.parametrize("valid", [
    "מה", "שה", "כשה", "וכשה", "שמה", "כשמה", "ומה", "משה", "ושמה", "וה", "וכש", "כשל", "כשב", "כשמ",
    "בה", "כה", "לה", "ובה", "וכה", "ולה", "שלה", "שבה", "שכה", "כשלה", "כשבה", "ושלה", "וכשלה",
])
def test_valid_prefixes_generated(valid):
    assert valid in DEFAULT_PREFIXES


def test_generated_set_contains_every_old_prefix():
    old = ("ב", "מ", "כ", "ו", "ה", "מה", "ל", "וב", "וה")  # old master DEFAULT_PREFIXES
    old_valid = old + ("וכ", "ול", "ומ", "וש", "מש", "שה", "שב", "שכ", "של", "שמ", "כש", "ש",
                       "ומה", "ומש", "ושה", "ושב", "ושל", "ושמ", "וכש", "כשה", "וכשה")
    for p in old_valid:
        assert p in DEFAULT_PREFIXES, p


# ---------------------------------------------------------------------------
# Prefix + abbreviation / word (one prefix set for everything)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", sorted(DEFAULT_PREFIXES))
def test_every_entry_with_every_prefix(prefix):
    """Every prefixable entry (abbreviations AND correction words) takes every prefix."""
    prefixable = {**RABBINIC_NIKKUD_DICT, **dict(additional_nikkud_corrections)}
    for key, value in prefixable.items():
        assert apply(f"א {prefix}{key} ב") == f"א {prefix}{value} ב", (prefix, key)
    for key in NO_PREFIX_NIKKUD_DICT:  # letter names never take a prefix
        if prefix:
            assert apply(f"סימן {prefix}{key} ב") == f"סימן {prefix}{key} ב", (prefix, key)


@pytest.mark.parametrize("word", ['מהרמב"ם', 'שהרמב"ם', 'כשהרמב"ם', 'וכשהרמב"ם', 'שמהרמב"ם',
                                  'ולרמב"ם', 'וברמב"ם', 'כשברמב"ם',
                                  'בהרמב"ם', 'להרמב"ם', 'כהרמב"ם', 'ובהרמב"ם', 'ולהרמב"ם',
                                  'וכהרמב"ם', 'שלהרמב"ם', 'שבהרמב"ם', 'כשלהרמב"ם',
                                  'להרשב"א', 'כהרשב"א', 'בהרא"ש'])
def test_prefix_examples_replaced_with_prefix_preserved(word):
    expected = word[: -len('רמב"ם')] + RAMBAM if word.endswith('רמב"ם') else None
    if expected is None:
        key = word[word.index("ה") + 1:]
        expected = word[: word.index("ה") + 1] + RABBINIC_NIKKUD_DICT[key]
    assert apply(word) == expected


@pytest.mark.parametrize("word, key, value", [
    ("בהשפתי", "שפתי", SHAFTEI), ("להשפתי", "שפתי", SHAFTEI), ("כהשפתי", "שפתי", SHAFTEI),
    ("ובהשפתי", "שפתי", SHAFTEI), ("שלהנבילה", "נבילה", NEVILA), ("כשלהנבילה", "נבילה", NEVILA),
    ("בשפתי", "שפתי", SHAFTEI), ("מהשפתי", "שפתי", SHAFTEI), ("ולנבילה", "נבילה", NEVILA),
])
def test_correction_words_use_the_same_prefix_set(word, key, value):
    assert apply(word) == word[: -len(key)] + value


@pytest.mark.parametrize("word", ['ארמב"ם', 'זהרמב"ם', 'בהה"רמב"ם', 'בהרמב"ם"ל', 'בהבשפתי', 'הבשפתי', 'לכהרמב"ם'])
def test_boundaries_and_invalid_combinations_still_not_matched(word):
    assert apply(word) == word


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


@pytest.mark.parametrize("prefix", ["ב", "ו", "ל", "מ", "ה", "כ", "ש", "וב", "וה", "ול", "מה", "שה", "וכש", "כש", "כשה", "וכשה", "שמ", "שמה", "כשב", "ומה"])
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
# Letter entries (הא / פא / צדי): standalone only, no prefixes, letter-name context only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    # after a context word (סימן, סעיף, ס"ק, דף, פרק, אות, סי')
    ("סימן הא", "סימן הֵא"),
    ("סימן פא", "סימן פֵּא"),
    ("סימן צדי", "סימן צָדִי"),
    ("סעיף הא", "סעיף הֵא"),
    ('ס"ק פא', 'ס"ק פֵּא'),
    ("ס״ק פא", "ס״ק פֵּא"),
    ("דף הא", "דף הֵא"),
    ("פרק צדי", "פרק צָדִי"),
    ("האות הא.", "האות הֵא."),
    ("אות פא,", "אות פֵּא,"),
    ("סי' הא", "סי' הֵא"),
    ("סי׳ הא", "סי׳ הֵא"),
    # context word with a prefix
    ("בסימן הא", "בסימן הֵא"),
    ("לסעיף פא", "לסעיף פֵּא"),
    ("ובאות צדי", "ובאות צָדִי"),
    ("מהסימן הא", "מהסימן הֵא"),
    # separators between context word and letter name: blanks, quotes, colon, dash, brackets
    ("סימן  \t הא", "סימן  \t הֵא"),
    ('סימן "הא"', 'סימן "הֵא"'),
    ("סימן 'פא'", "סימן 'פֵּא'"),
    ("סימן: הא", "סימן: הֵא"),
    ("סימן - הא", "סימן - הֵא"),
    ("סימן־הא", "סימן־הֵא"),
    ("סימן (הא)", "סימן (הֵא)"),
    ("(סימן צדי)", "(סימן צָדִי)"),
    # directly adjacent to another letter-name word (chains)
    ("סימן צדי הא", "סימן צָדִי הֵא"),
    ("סימן פא זין", "סימן פֵּא זין"),
    ("צדי הא", "צָדִי הֵא"),
    ("צדיק הא", "צדיק הֵא"),
    ("הא ריש", "הֵא ריש"),
    ("פא זין", "פֵּא זין"),
    ("קוף הא", "קוף הֵא"),
    ("הא\tוו", "הֵא\tוו"),
    ("סימן פא הא", "סימן פֵּא הֵא"),
    # next to punctuation / start / end / newline, with context
    ("שורה\nסימן הא", "שורה\nסימן הֵא"),
    ("אות הא\nוהלאה", "אות הֵא\nוהלאה"),
    ("סימן צדי\nסימן הא", "סימן צָדִי\nסימן הֵא"),
    ("סימן הא-פא", "סימן הֵא-פא"),   # hyphen is not blank space: פא has no context
    ("סימן הא, פא וצדי", "סימן הֵא, פא וצדי"),
])
def test_letter_entries_in_letter_name_context(text, expected):
    assert apply(text) == expected


@pytest.mark.parametrize("text", [
    # ordinary words: Aramaic הא, sentence start, no context
    "הא", "פא", "צדי",
    "הא דתניא", 'הא קמ"ל', "הא כיצד", "הא למדת", "ואמר הא דתניא",
    "הא. פא! צדי?",
    "(הא)", '"צדי"', "הא-פא", "הא\nפא",
    # context word on a different line, or separated by a sentence stop / another word
    "סימן\nהא", "סימן. הא", "סימן, הא", "סימן? הא", "סימן; הא", "סימן אחד הא",
    "ראה סימן צ\"ה הא דתניא",
    # context word must be a whole word (a valid prefix is allowed, nothing else)
    "אסימן הא", "ססימן הא", "סימנים הא", "דפים הא",
    # letter names never take a prefix
    "בהא", "והא", "להא", "מהא", "שהא", "הפא", "בפא", "וצדי", "בצדי", "מצדי",
    "סימן בהא", "סימן והא", "סימן ופא", "סימן הצדי",
    # inside longer words
    "האדם", "פאה", "צדיק", "צדיקים", "ופא", "סימן האדם", "סימן צדיקים",
    # a neighbour must be a whole letter-name word
    "הא צדיקים", "הא בית-דין", "הא זינוק", "אלפים הא", "בית-דין הא",
])
def test_letter_entries_not_replaced_without_letter_name_context(text):
    assert apply(text) == text


@pytest.mark.parametrize("text", [
    "סימן הא'", "סימן הא׳", "אות הא' וכו'", "סעיף פא'", "סעיף צדי׳.", "סי' הא'",
    "הא' פא'", "סימן הא' פא'",
])
def test_trailing_geresh_is_not_a_boundary_for_letter_names(text):
    """הא' is an ordinal ("the first") - never the letter name, even right after סימן."""
    assert apply(text) == text


def test_geresh_wrapped_letter_name_is_still_a_letter_name():
    assert apply("סימן 'הא'") == "סימן 'הֵא'"
    assert apply("סימן ׳הא׳") == "סימן ׳הֵא׳"


def test_letter_name_context_is_per_match():
    text = "הא דתניא. בסימן צדי הא נאמר: הא קמ\"ל, ובדף פא."
    assert apply(text) == (
        "הא דתניא. בסימן "
        + NO_PREFIX_NIKKUD_DICT["צדי"] + " " + NO_PREFIX_NIKKUD_DICT["הא"]
        + " נאמר: הא קמ\"ל, ובדף " + NO_PREFIX_NIKKUD_DICT["פא"] + "."
    )


def test_siman_numbers_spoken_as_letter_names():
    """The pipeline writes simanim as letter names: 94 -> צדי דלת, 95 -> צדי הא, 85 -> פא הא."""
    assert apply("סימן צדי דלת") == f"סימן {NO_PREFIX_NIKKUD_DICT['צדי']} דלת"
    assert apply("סימן צדי הא") == f"סימן {NO_PREFIX_NIKKUD_DICT['צדי']} {NO_PREFIX_NIKKUD_DICT['הא']}"
    assert apply("סימן פא הא") == f"סימן {NO_PREFIX_NIKKUD_DICT['פא']} {NO_PREFIX_NIKKUD_DICT['הא']}"
    assert apply("סימן פא זין") == f"סימן {NO_PREFIX_NIKKUD_DICT['פא']} זין"


def test_letter_name_context_is_linear_on_long_lines():
    import time
    text = "הא דתניא " * 40000          # one 360K-char line, no context anywhere
    start = time.perf_counter()
    assert apply(text) == text
    assert time.perf_counter() - start < 5


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
    text = "\n".join(f"הנה סימן {key} כאן, ו{key}." for key in entries)
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
        assert apply(f"סימן {key} ב") == f"סימן {value} ב", key
        assert apply(f"סימן ב{key} ב") == f"סימן ב{key} ב", key
        assert apply(f"א {key} ב") == f"א {key} ב", key  # no letter-name context


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
