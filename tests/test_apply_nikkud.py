"""Tests for pipeline/apply_nikkud.py (regex word-boundary matching with Hebrew prefixes)."""
import pytest

from pipeline import apply_nikkud as an
from pipeline.apply_nikkud import (
    DEFAULT_PREFIXES,
    NO_PREFIX_NIKKUD_DICT,
    RABBINIC_NIKKUD_DICT,
    additional_nikkud_corrections,
    apply_nikkud_to_abbreviations as apply,
)

# Expected values come from the module's own tables (avoids Unicode mark-order typos).
SHAFTEI = dict(additional_nikkud_corrections)["שפתי כהן"]  # multi-word entry
NEVILA = dict(additional_nikkud_corrections)["נבילה"]
RAMBAM = RABBINIC_NIKKUD_DICT['רמב"ם']


# ---------------------------------------------------------------------------
# Public API still importable / working
# ---------------------------------------------------------------------------

def test_public_api_importable():
    """Live public surface: dicts + prefixes + apply; no legacy expanders."""
    assert callable(apply)
    assert additional_nikkud_corrections and RABBINIC_NIKKUD_DICT and NO_PREFIX_NIKKUD_DICT
    assert DEFAULT_PREFIXES and DEFAULT_PREFIXES[0] == ""
    assert apply('רמב"ם') == RAMBAM
    assert apply("שפתי כהן") == SHAFTEI
    assert apply("סימן הא") == f"סימן {NO_PREFIX_NIKKUD_DICT['הא']}"
    for name in (
        "boundary_variants",
        "prefixed_variants",
        "expand_correction_pair",
        "build_additional_nikkud_map",
        "build_full_nikkud_map",
        "DEFAULT_TRAILING_PUNCT",
    ):
        assert not hasattr(an, name), name


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


# --- nikkud-agnostic matching: the text may already be vocalized (the polish vocalizes it) ---

def _fake_vocalize(word, point="\u05B7"):
    """word with `point` (patah by default) after every Hebrew letter: any vocalization must match."""
    return "".join(ch + point if "\u05D0" <= ch <= "\u05EA" else ch for ch in word)


@pytest.mark.parametrize("prefix", sorted(DEFAULT_PREFIXES))
@pytest.mark.parametrize("vocalized_prefix", [False, True])
@pytest.mark.parametrize("vocalized_word", [False, True])
def test_every_entry_with_every_prefix_ignores_nikkud(prefix, vocalized_prefix, vocalized_word):
    """Every generated prefix, with and without nikkud on the prefix and on the word: the entry's
    value replaces the word, and the prefix is kept exactly as written (its nikkud included)."""
    pre = _fake_vocalize(prefix, "\u05B4") if vocalized_prefix else prefix
    prefixable = {**RABBINIC_NIKKUD_DICT, **dict(additional_nikkud_corrections)}
    for key, value in prefixable.items():
        word = _fake_vocalize(key) if vocalized_word else key
        assert apply(f"א {pre}{word} ב") == f"א {pre}{value} ב", (pre, word)
    for key, value in NO_PREFIX_NIKKUD_DICT.items():  # letter names: bare only, any nikkud
        word = _fake_vocalize(key) if vocalized_word else key
        expected = f"סימן {value} ב" if not prefix else f"סימן {pre}{word} ב"
        assert apply(f"סימן {pre}{word} ב") == expected, (pre, word)


KAHAL = dict(additional_nikkud_corrections)["כחל"]


@pytest.mark.parametrize("text, expected", [
    ("כֹּחַל", KAHAL),                                   # the model's other reading: dictionary wins
    ("כְּחָל", KAHAL),                                    # already the dictionary's
    ("וּבִכְחָל", "וּבִ" + KAHAL),                          # vocalized prefix kept as written
    ("ובכֹחַל", "וב" + KAHAL),
    ("הָרַמְבַּ\"ם", "הָ" + RAMBAM),                       # vocalized abbreviation, ASCII quote
    ("לָרַמְבַּ״ם", "לָ" + RAMBAM),                         # gershayim
    ("וּמֵהָרַמְבַּ\"ם", "וּמֵהָ" + RAMBAM),
    ("רַמְבַּם", RAMBAM),                                  # unquoted abbreviation, vocalized
    ("סימן צָדִי הֵא", f"סימן {NO_PREFIX_NIKKUD_DICT['צדי']} {NO_PREFIX_NIKKUD_DICT['הא']}"),
    ("סִימָן הָא", f"סִימָן {NO_PREFIX_NIKKUD_DICT['הא']}"),  # vocalized context word
    ("צָדִי פֵא", f"{NO_PREFIX_NIKKUD_DICT['צדי']} {NO_PREFIX_NIKKUD_DICT['פא']}"),  # adjacent names
])
def test_vocalized_match_gets_the_dictionary_nikkud(text, expected):
    assert apply(text) == expected
    assert apply(expected) == expected


@pytest.mark.parametrize("text", [
    "כְּחָלִים", "הַכֹּחַלִית", "קְכֹחַל",        # longer word / not a prefix
    "הָא דְּתַנְיָא", "הֵא לָךְ", "בְּהֵא",          # letter name outside its context / with a prefix
    "הָבְּשִׂפְתֵי כהן",                          # invalid prefix combination
    "רַמְבַּ\"מִים",                              # quote glued inside a longer word
    "שִׁלּוּחַ", "סֵפֶר",                         # not in the dictionary: the model's nikkud stays
])
def test_vocalized_non_matches_untouched(text):
    assert apply(text) == text


def test_overrides_report_only_vocalized_words_that_changed():
    text = 'כחל וּבִכֹחַל\nהָרַמְבַּ"ם וכְּחָל סֵפֶר'
    out, overrides = an.apply_nikkud_with_overrides(text)
    assert out == f'{KAHAL} וּבִ{KAHAL}\nהָ{RAMBAM} ו{KAHAL} סֵפֶר'
    # plain כחל was only vocalized; וכְּחָל already had the dictionary's nikkud: neither is an override
    assert [(o.model, o.dictionary) for o in overrides] == [("וּבִכֹחַל", "וּבִ" + KAHAL),
                                                            ('הָרַמְבַּ"ם', "הָ" + RAMBAM)]
    for o in overrides:  # start is the replacement's offset in the output text
        assert out[o.start:o.start + len(o.dictionary)] == o.dictionary


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
    ("בהשפתי כהן", "שפתי כהן", SHAFTEI), ("להשפתי כהן", "שפתי כהן", SHAFTEI), ("כהשפתי כהן", "שפתי כהן", SHAFTEI),
    ("ובהשפתי כהן", "שפתי כהן", SHAFTEI), ("שלהנבילה", "נבילה", NEVILA), ("כשלהנבילה", "נבילה", NEVILA),
    ("בשפתי כהן", "שפתי כהן", SHAFTEI), ("מהשפתי כהן", "שפתי כהן", SHAFTEI), ("ולנבילה", "נבילה", NEVILA),
])
def test_correction_words_use_the_same_prefix_set(word, key, value):
    assert apply(word) == word[: -len(key)] + value


@pytest.mark.parametrize("word", ['ארמב"ם', 'זהרמב"ם', 'בהה"רמב"ם', 'בהרמב"ם"ל', 'בהבשפתי כהן', 'הבשפתי כהן', 'לכהרמב"ם'])
def test_boundaries_and_invalid_combinations_still_not_matched(word):
    assert apply(word) == word


# ---------------------------------------------------------------------------
# Boundaries for a multi-word additional correction entry (שפתי כהן)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("זה שפתי כהן", f"זה {SHAFTEI}"),                     # end of text
    ("זה שפתי כהן אומר", f"זה {SHAFTEI} אומר"),           # mid-sentence
    ("שפתי כהן אומר", f"{SHAFTEI} אומר"),                 # start of text
    ("שורה ראשונה\nשפתי כהן", f"שורה ראשונה\n{SHAFTEI}"),  # start of line
    ("אמר שפתי כהן\nוהלאה", f"אמר {SHAFTEI}\nוהלאה"),     # end of line
    ("שפתי כהן.", f"{SHAFTEI}."),
    ("שפתי כהן,", f"{SHAFTEI},"),
    ("שפתי כהן;", f"{SHAFTEI};"),
    ("שפתי כהן:", f"{SHAFTEI}:"),
    ("שפתי כהן!", f"{SHAFTEI}!"),
    ("שפתי כהן?", f"{SHAFTEI}?"),
    ("(שפתי כהן)", f"({SHAFTEI})"),
    ("[שפתי כהן]", f"[{SHAFTEI}]"),
    ('"שפתי כהן"', f'"{SHAFTEI}"'),
    ("'שפתי כהן'", f"'{SHAFTEI}'"),
    ('אמר "שפתי כהן" שם', f'אמר "{SHAFTEI}" שם'),
    ("שפתי כהן-שם", f"{SHAFTEI}-שם"),
    ("שם-שפתי כהן", f"שם-{SHAFTEI}"),
    ("שפתי כהן שפתי כהן", f"{SHAFTEI} {SHAFTEI}"),
    ("שפתי כהן\nשפתי כהן", f"{SHAFTEI}\n{SHAFTEI}"),
    # the space inside the key matches a run of spaces / tabs / NBSP
    ("שפתי  כהן", SHAFTEI),
    ("שפתי\tכהן", SHAFTEI),
    ("שפתי\u00A0כהן", SHAFTEI),
])
def test_boundaries(text, expected):
    assert apply(text) == expected


@pytest.mark.parametrize("prefix", ["ב", "ו", "ל", "מ", "ה", "כ", "ש", "וב", "וה", "ול", "מה", "שה", "וכש", "כש", "כשה", "וכשה", "שמ", "שמה", "כשב", "ומה"])
def test_prefixes_are_preserved(prefix):
    assert apply(f"אמר {prefix}שפתי כהן") == f"אמר {prefix}{SHAFTEI}"
    assert apply(f"{prefix}שפתי כהן") == f"{prefix}{SHAFTEI}"
    assert apply(f"{prefix}נבילה.") == f"{prefix}{NEVILA}."


def test_specific_prefix_examples():
    assert apply("בשפתי כהן") == "ב" + SHAFTEI
    assert apply("ושפתי כהן") == "ו" + SHAFTEI
    assert apply("ולשפתי כהן") == "ול" + SHAFTEI
    assert apply("ששפתי כהן") == "ש" + SHAFTEI
    assert apply("מהשפתי כהן") == "מה" + SHAFTEI


def test_placenta_spelling_variants_share_one_vocalization():
    assert apply("שליה") == apply("שלייה") == "שִׁלְיָה"
    assert apply("בשלייה") == "בשִׁלְיָה"


def test_udder_gets_its_dictionary_nikkud_with_prefixes():
    assert apply("כחל") == "כְּחָל"
    # the prefix is kept as written: ב + כְּחָל (read bi-khchal; the dagesh is the bare word's)
    assert apply("בכחל של בהמה") == "בכְּחָל של בהמה"
    assert apply("והכחל") == "והכְּחָל"
    assert apply("כחלים") == "כחלים"  # never inside a longer word


@pytest.mark.parametrize("text", [
    "הבשפתי כהן",   # not a valid prefix combination
    "שפתי כהנים",   # longer word
    "ישפתי כהן",
    "גשפתי כהן",
    "שפתי כהן\"ם",  # quote glued to the following letter = inside a word
    "ה\"שפתי כהן",  # quote glued to the preceding letter
    "שפתי",         # the bare first word is no longer an entry
    "שפתי\nכהן",    # a key space never matches a line break
    "נבילות",
    "קנבילה",
])
def test_not_replaced_inside_longer_word(text):
    assert apply(text) == text


def test_already_nikkuded_entry_value_unchanged():
    assert apply(SHAFTEI) == SHAFTEI
    assert apply("בשִׂפְתֵי כהן") == "ב" + SHAFTEI


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
    ('שו"ת', 'שׁוּת'),
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
# Letter entries (הא / הי / וו / פא / צדי, and the wrong form הֵא): standalone only, no prefixes,
# letter-name context only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    # after a context word (סימן, סעיף, ס"ק, דף, פרק, אות, סי')
    ("סימן הא", "סימן הֵי"),
    ("סימן פא", "סימן פֵּא"),
    ("סימן צדי", "סימן צָדִי"),
    ("סעיף הא", "סעיף הֵי"),
    ('ס"ק פא', 'ס"ק פֵּא'),
    ("ס״ק פא", "ס״ק פֵּא"),
    ("דף הא", "דף הֵי"),
    ("פרק צדי", "פרק צָדִי"),
    ("האות הא.", "האות הֵי."),
    ("אות פא,", "אות פֵּא,"),
    ("סי' הא", "סי' הֵי"),
    ("סי׳ הא", "סי׳ הֵי"),
    # context word with a prefix
    ("בסימן הא", "בסימן הֵי"),
    ("לסעיף פא", "לסעיף פֵּא"),
    ("ובאות צדי", "ובאות צָדִי"),
    ("מהסימן הא", "מהסימן הֵי"),
    # separators between context word and letter name: blanks, quotes, colon, dash, brackets
    ("סימן  \t הא", "סימן  \t הֵי"),
    ('סימן "הא"', 'סימן "הֵי"'),
    ("סימן 'פא'", "סימן 'פֵּא'"),
    ("סימן: הא", "סימן: הֵי"),
    ("סימן - הא", "סימן - הֵי"),
    ("סימן־הא", "סימן־הֵי"),
    ("סימן (הא)", "סימן (הֵי)"),
    ("(סימן צדי)", "(סימן צָדִי)"),
    # directly adjacent to another letter-name word (chains)
    ("סימן צדי הא", "סימן צָדִי הֵי"),
    ("סימן פא זין", "סימן פֵּא זין"),
    ("צדי הא", "צָדִי הֵי"),
    ("צדיק הא", "צדיק הֵי"),
    ("הא ריש", "הֵי ריש"),
    ("פא זין", "פֵּא זין"),
    ("קוף הא", "קוף הֵי"),
    ("הא\tוו", "הֵי\tוָו"),
    ("סימן פא הא", "סימן פֵּא הֵי"),
    # next to punctuation / start / end / newline, with context
    ("שורה\nסימן הא", "שורה\nסימן הֵי"),
    ("אות הא\nוהלאה", "אות הֵי\nוהלאה"),
    ("סימן צדי\nסימן הא", "סימן צָדִי\nסימן הֵי"),
    ("סימן הא-פא", "סימן הֵי-פא"),   # hyphen is not blank space: פא has no context
    ("סימן הא, פא וצדי", "סימן הֵי, פא וצדי"),
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
    assert apply("סימן 'הא'") == "סימן 'הֵי'"
    assert apply("סימן ׳הא׳") == "סימן ׳הֵי׳"


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


HEI = NO_PREFIX_NIKKUD_DICT["הא"]


@pytest.mark.parametrize("text, expected", [
    # the letter ה written as הי, or wrongly vocalized הֵא (e.g. by the polish), is respelled הֵי
    ("סעיף הי", "סעיף הֵי"),
    ("דף הי עמוד בית", "דף הֵי עמוד בית"),
    ("סימן יוד הי", "סימן יוד הֵי"),
    ("סעיף הֵא", "סעיף הֵי"),
    ("סעיף הֵא.", "סעיף הֵי."),
    # the polish output that showed the bug: a wrongly vocalized neighbour still gives context
    ("צדי אלף עד צדי הֵא", "צָדִי אלף עד צָדִי הֵי"),
    ("בדף למד הֵא עמוד בית", "בדף למד הֵי עמוד בית"),
    ("סימן קוף הֵא", "סימן קוף הֵי"),
    # וו as a letter name
    ("סעיף וו", "סעיף וָו"),
    ("דף קוף טית וו", "דף קוף טית וָו"),
])
def test_he_and_vav_letter_names(text, expected):
    assert apply(text) == expected
    assert apply(expected) == expected


@pytest.mark.parametrize("text", [
    "הי מינייהו", "הי ניהו", "הֵא לך", "הֵא", "וו", "ווי העמודים", "סימן ווי",
    "בהי", "בהֵא", "סימן בהֵא", "סימן והי",   # never with a prefix
    "סעיף הֵא'", "סעיף הי'",                  # a trailing geresh is an ordinal
    "סעיף הֵי",                                # already correct
])
def test_he_and_vav_untouched_outside_letter_name_context(text):
    assert apply(text) == text


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


# ---------------------------------------------------------------------------
# Multi-word keys: a space in a key matches a run of spaces / tabs / NBSP, never a line break
# ---------------------------------------------------------------------------

BEIT_YOSEF_NIKKUD = "בֵּית יוֹסֵף"  # written with ONE ASCII space, as in the dict
EVEN_HAEZER = RABBINIC_NIKKUD_DICT["אבן העזר"]


@pytest.fixture
def beit_yosef(monkeypatch):
    """Adds the multi-word entry ("בית יוסף", ...) to additional_nikkud_corrections for one test."""
    an._build_matcher.cache_clear()
    monkeypatch.setattr(
        an,
        "additional_nikkud_corrections",
        list(additional_nikkud_corrections) + [("בית יוסף", BEIT_YOSEF_NIKKUD)],
    )
    yield apply
    an._build_matcher.cache_clear()


@pytest.mark.parametrize(
    "gap",
    [" ", "  ", "     ", "\t", "\u00A0", "\u00A0\u00A0", " \t\u00A0 "],
    ids=["space", "2spaces", "5spaces", "tab", "nbsp", "2nbsp", "mixed"],
)
def test_multiword_key_matches_any_run_of_blanks(beit_yosef, gap):
    assert beit_yosef(f"בית{gap}יוסף") == BEIT_YOSEF_NIKKUD
    assert beit_yosef(f"א בית{gap}יוסף ב") == f"א {BEIT_YOSEF_NIKKUD} ב"


@pytest.mark.parametrize("prefix", ["ה", "ב", "וב", "ל", "מה", "ושה", "כשב"])
@pytest.mark.parametrize("gap", [" ", "  ", "\t", "\u00A0"], ids=["space", "2spaces", "tab", "nbsp"])
def test_multiword_key_with_prefixes(beit_yosef, prefix, gap):
    assert beit_yosef(f"{prefix}בית{gap}יוסף") == prefix + BEIT_YOSEF_NIKKUD
    assert beit_yosef(f"ראה {prefix}בית{gap}יוסף כאן") == f"ראה {prefix}{BEIT_YOSEF_NIKKUD} כאן"


@pytest.mark.parametrize("punct", [".", ",", "!", "?", ":", ";", ")", '"', "-"])
def test_multiword_key_with_trailing_punctuation(beit_yosef, punct):
    assert beit_yosef(f"בית  יוסף{punct}") == BEIT_YOSEF_NIKKUD + punct
    assert beit_yosef(f"וב(בית\u00A0יוסף{punct} אחר") == f"וב({BEIT_YOSEF_NIKKUD}{punct} אחר"


def test_multiword_key_at_text_and_line_edges(beit_yosef):
    assert beit_yosef("בית  יוסף") == BEIT_YOSEF_NIKKUD
    assert beit_yosef("שורה\nבית  יוסף\nשורה") == f"שורה\n{BEIT_YOSEF_NIKKUD}\nשורה"
    assert beit_yosef("(בית\tיוסף)") == f"({BEIT_YOSEF_NIKKUD})"


def test_multiword_key_output_is_the_value_as_written(beit_yosef):
    out = beit_yosef("בית \t\u00A0 יוסף")
    assert out == BEIT_YOSEF_NIKKUD
    assert out.count(" ") == 1 and "\t" not in out and "\u00A0" not in out


@pytest.mark.parametrize("brk", ["\n", "\r", "\r\n", "\n\n", " \n ", "\u2028", "\u2029"])
def test_multiword_key_does_not_match_across_a_line_break(beit_yosef, brk):
    text = f"בית{brk}יוסף"
    assert beit_yosef(text) == text
    text = f"הבית{brk}יוסף וגם ובבית{brk}יוסף"
    assert beit_yosef(text) == text


@pytest.mark.parametrize("text", ["בית-יוסף", "בית\u05BEיוסף", "ביתיוסף", "בית.יוסף", "בית,יוסף", "בית־יוסף", "בית_יוסף"])
def test_multiword_key_does_not_match_hyphen_glued_or_punctuation(beit_yosef, text):
    assert beit_yosef(text) == text


@pytest.mark.parametrize("text", ["אבית יוסף", "בית יוספי", "בית יוסףא", "בית יוסף'ם"])
def test_multiword_key_still_needs_word_boundaries(beit_yosef, text):
    assert beit_yosef(text) == text


def test_multiword_key_matches_inside_a_sentence_and_repeats(beit_yosef):
    text = "כתב בית  יוסף וגם ב\tבית יוסף, וה\u00A0בית יוסף."
    # "ב\tבית יוסף" -> ב is a separate word, "ה\u00A0בית יוסף" -> ה is a separate word
    assert beit_yosef(text) == (
        f"כתב {BEIT_YOSEF_NIKKUD} וגם ב\t{BEIT_YOSEF_NIKKUD}, וה\u00A0{BEIT_YOSEF_NIKKUD}."
    )


def test_single_word_entries_still_work_next_to_multiword_entry(beit_yosef):
    assert beit_yosef("נבילה") == NEVILA
    assert beit_yosef("בנבילה נבילה") == "ב" + NEVILA + " " + NEVILA
    assert beit_yosef('הרמב"ם') == "ה" + RAMBAM
    assert beit_yosef('ב"י') == RABBINIC_NIKKUD_DICT['ב"י']
    assert beit_yosef("בית") == "בית"  # first word of the key alone is not an entry
    assert beit_yosef("יוסף") == "יוסף"
    # the multi-word entry and the single-word ones in one text, mixed gaps
    assert beit_yosef("שפתי  כהן ובית\tיוסף נבילה") == f"{SHAFTEI} ו{BEIT_YOSEF_NIKKUD} {NEVILA}"


def test_multiword_entry_is_idempotent(beit_yosef):
    once = beit_yosef("הבית  יוסף, בית\tיוסף ובבית\u00A0יוסף.")
    assert once != "הבית  יוסף, בית\tיוסף ובבית\u00A0יוסף."
    assert beit_yosef(once) == once


def test_multiword_entry_in_rabbinic_dict_is_used_too(monkeypatch):
    an._build_matcher.cache_clear()
    monkeypatch.setitem(RABBINIC_NIKKUD_DICT, "בית יוסף", BEIT_YOSEF_NIKKUD)
    # the real additional_nikkud_corrections now has its own "בית יוסף"; drop it so the
    # rabbinic-dict entry is the one under test
    monkeypatch.setattr(an, "additional_nikkud_corrections",
                        [(k, v) for k, v in additional_nikkud_corrections if k != "בית יוסף"])
    try:
        assert apply("בית  יוסף") == BEIT_YOSEF_NIKKUD
        assert apply("בית\u00A0יוסף\tנבילה") == f"{BEIT_YOSEF_NIKKUD}\t{NEVILA}"
        assert apply("בית\nיוסף") == "בית\nיוסף"
    finally:
        an._build_matcher.cache_clear()


def test_multiword_key_with_quote_variants_and_spaces(monkeypatch):
    """Quote canonicalization and blank collapsing work together in one key."""
    an._build_matcher.cache_clear()
    monkeypatch.setattr(an, "additional_nikkud_corrections", [('מהר"ם שיף', "מַהֲרַם שִׁיף")])
    try:
        assert apply('מהר"ם  שיף') == "מַהֲרַם שִׁיף"
        assert apply("מהר\u05F4ם\u00A0שיף") == "מַהֲרַם שִׁיף"
        assert apply('למהר"ם\tשיף') == "ל" + "מַהֲרַם שִׁיף"
        assert apply('מהר"ם\nשיף') == apply('מהר"ם') + "\nשיף"
    finally:
        an._build_matcher.cache_clear()


def test_key_with_extra_blanks_in_the_dict_is_normalized(monkeypatch):
    an._build_matcher.cache_clear()
    monkeypatch.setattr(an, "additional_nikkud_corrections", [("בית \u00A0 יוסף", BEIT_YOSEF_NIKKUD)])
    try:
        assert apply("בית יוסף") == BEIT_YOSEF_NIKKUD
        assert apply("בית\t\tיוסף") == BEIT_YOSEF_NIKKUD
    finally:
        an._build_matcher.cache_clear()


# --- the real multi-word entry in RABBINIC_NIKKUD_DICT -----------------------

def test_even_haezer_is_a_real_multiword_entry():
    assert "אבן העזר" in RABBINIC_NIKKUD_DICT
    assert apply("אבן העזר") == EVEN_HAEZER


@pytest.mark.parametrize("gap", ["  ", "   ", "\t", "\u00A0", " \u00A0"], ids=["2sp", "3sp", "tab", "nbsp", "mixed"])
def test_even_haezer_with_extra_blanks(gap):
    assert apply(f"אבן{gap}העזר") == EVEN_HAEZER
    assert apply(f"באבן{gap}העזר.") == "ב" + EVEN_HAEZER + "."
    assert apply(f"ראה ובאבן{gap}העזר, סימן") == f"ראה ו" + "ב" + EVEN_HAEZER + ", סימן"


@pytest.mark.parametrize("text", ["אבן\nהעזר", "אבן\r\nהעזר", "אבן-העזר", "אבןהעזר"])
def test_even_haezer_not_matched_across_break_hyphen_or_glued(text):
    assert apply(text) == text


def test_context_regexes_and_letter_names_unaffected_by_blank_rule():
    # none of the context words / letter names contains a space, so nothing changes for them
    assert all(" " not in w for w in an.LETTER_NAME_CONTEXT_WORDS)
    assert all(" " not in w for w in an.LETTER_NAMES)
    assert all(" " not in k for k in NO_PREFIX_NIKKUD_DICT)
    an._letter_context_regexes.cache_clear()
    an._letter_context_regexes(tuple(DEFAULT_PREFIXES))  # compiles
    assert apply("סימן צדי הא") == f"סימן {NO_PREFIX_NIKKUD_DICT['צדי']} {NO_PREFIX_NIKKUD_DICT['הא']}"
    assert apply("סימן  \t צדי") == f"סימן  \t {NO_PREFIX_NIKKUD_DICT['צדי']}"
    assert apply("סימן\nצדי") == "סימן\nצדי"
    assert an._key_to_regex("סי'") == f"ס{an._ANY_POINTS}י{an._ANY_POINTS}['\u05F3]"
    assert an._key_to_regex("a b") == f"a{an._BLANK}+b"
    assert an._norm("אבן \t\u00A0 העזר\u05F4") == 'אבן העזר"'


# ---------------------------------------------------------------------------
# Unquoted abbreviations (gershayim dropped by the polish stage)
# ---------------------------------------------------------------------------

MAHARSHAL = RABBINIC_NIKKUD_DICT['מהרש"ל']


@pytest.mark.parametrize("text, expected", [
    ("מהרשל", MAHARSHAL),
    ("והמהרשל אומר", f"וה{MAHARSHAL} אומר"),
    ("כדברי הרמבם.", f"כדברי ה{RAMBAM}."),
    ("ולדעת הרשבא", "ולדעת ה" + RABBINIC_NIKKUD_DICT['רשב"א']),
    ("שהמגא כתב", "שה" + RABBINIC_NIKKUD_DICT['מג"א'] + " כתב"),
    ("מהמהרשא", "מה" + RABBINIC_NIKKUD_DICT['מהרש"א']),
    ("(הסמע)", "(ה" + RABBINIC_NIKKUD_DICT['סמ"ע'] + ")"),
])
def test_unquoted_abbreviation_with_and_without_prefix(text, expected):
    assert apply(text) == expected


def test_unquoted_matches_like_quoted():
    for key in an.UNQUOTED_ABBREVIATIONS:
        unquoted = key.replace('"', "")
        for prefix in ("", "ו", "וה", "כשה", "מ"):
            assert apply(prefix + unquoted) == apply(prefix + key) == prefix + RABBINIC_NIKKUD_DICT[key]


@pytest.mark.parametrize("text", [
    "ראש השנה", "הוא רזה", "רמה גבוהה", "רמא ליה", "משך הזמן", "מרשי ושרשי הדברים", "הרשי",
    "הנציב", "גרא", "תשבץ", "סמק", "ריף", "רן", "רת", "טז", "בח", "שבח", "בי", "נט", "עפ",
    "יוד", "אוח", "חנן", "שוע", "שות", "ושות",
])
def test_ambiguous_unquoted_forms_unchanged(text):
    assert apply(text) == text


def test_unquoted_inside_longer_word_or_after_quote_not_matched():
    assert apply("רמבםים") == "רמבםים"
    assert apply('ה"רמבם') == 'ה"רמבם'  # quote glued to a previous letter: inside a word


def test_unquoted_allowlist_entries_are_quoted_dict_keys(monkeypatch):
    assert all('"' in k and k in RABBINIC_NIKKUD_DICT for k in an.UNQUOTED_ABBREVIATIONS)
    monkeypatch.setattr(an, "UNQUOTED_ABBREVIATIONS", ("רמבם",))
    with pytest.raises(ValueError, match="not a quoted RABBINIC_NIKKUD_DICT key"):
        an._unquoted_variants()


def test_unquoted_output_is_idempotent():
    out = apply("והמהרשל והרמבם")
    assert apply(out) == out
