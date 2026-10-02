import pytest
from pipeline.gematria import int_to_gematria, gematria_to_int

def test_single_digit_gematria():
    assert int_to_gematria(1) == "א'"
    assert int_to_gematria(2) == "ב'"
    assert int_to_gematria(9) == "ט'"

def test_tens_gematria():
    assert int_to_gematria(10) == "י'"
    assert int_to_gematria(20) == "כ'"
    assert int_to_gematria(90) == "צ'"

def test_double_digits_gematria():
    assert int_to_gematria(94) == 'צ"ד'
    assert int_to_gematria(42) == 'מ"ב'

def test_special_combinations():
    # 15 and 16 should be ט"ו and ט"ז instead of יה and יו
    assert int_to_gematria(15) == 'ט"ו'
    assert int_to_gematria(16) == 'ט"ז'
    # 115 and 116 should also be modified
    assert int_to_gematria(115) == 'קט"ו'
    assert int_to_gematria(116) == 'קט"ז'

def test_hundreds_gematria():
    assert int_to_gematria(100) == "ק'"
    assert int_to_gematria(101) == 'ק"א'
    assert int_to_gematria(300) == "ש'"
    assert int_to_gematria(342) == 'שמ"ב'

def test_invalid_range_raises_error():
    with pytest.raises(ValueError):
        int_to_gematria(0)
    with pytest.raises(ValueError):
        int_to_gematria(-5)
    with pytest.raises(ValueError):
        int_to_gematria(1000)

def test_gematria_to_int():
    assert gematria_to_int("א") == 1
    assert gematria_to_int("א'") == 1
    assert gematria_to_int("ב") == 2
    assert gematria_to_int("ט'") == 9
    assert gematria_to_int("י") == 10
    assert gematria_to_int("י'") == 10
    assert gematria_to_int("יא") == 11
    assert gematria_to_int('י"א') == 11
    assert gematria_to_int('י״א') == 11
    assert gematria_to_int("י'א") == 11
    assert gematria_to_int('ט"ו') == 15
    assert gematria_to_int('ט"ז') == 16
    assert gematria_to_int('כ"ב') == 22
    assert gematria_to_int('צ"ד') == 94
    assert gematria_to_int('11') == 11
    assert gematria_to_int('1') == 1
    assert gematria_to_int('') is None
    assert gematria_to_int('0') is None
    assert gematria_to_int('invalid') is None


# --- spoken (TTS) letter names ------------------------------------------------

import re as _re
from pipeline.gematria import LETTER_NAMES, int_to_spoken_gematria

_POINTS = _re.compile("[\u0591-\u05C7]")


def test_letter_names_cover_every_letter_int_to_gematria_can_emit():
    letters = set()
    for n in range(1, 1000):
        letters |= set(ch for ch in int_to_gematria(n) if ch not in "\"'")
    assert letters <= set(LETTER_NAMES)


def test_spoken_gematria_matches_written_letters():
    for n in range(1, 1000):
        written = [ch for ch in int_to_gematria(n) if ch in LETTER_NAMES]
        spoken = int_to_spoken_gematria(n).split(" ")
        assert len(spoken) == len(written) > 0
        assert spoken == [LETTER_NAMES[ch] for ch in written]


def test_spoken_gematria_siman_prompt_convention():
    # The prompt spells simanim as "צדי דלת" (not "צדיק"): vowel points are only an aid.
    plain = lambda n: _POINTS.sub("", int_to_spoken_gematria(n))
    assert plain(94) == "צדי דלת"
    assert plain(95) == "צדי הא"
    assert plain(96) == "צדי וו"
    assert plain(97) == "צדי זין"
    assert plain(15) == "טית וו" and plain(16) == "טית זין"   # ט"ו / ט"ז, never יה / יו


def test_spoken_gematria_only_adds_vowel_points_to_ambiguous_letters():
    for letter, name in LETTER_NAMES.items():
        if _POINTS.search(name):
            assert letter in "הופצ"


def test_spoken_gematria_rejects_out_of_range():
    for bad in (0, 1000, -3):
        with pytest.raises(ValueError):
            int_to_spoken_gematria(bad)


def test_spoken_letter_names_match_apply_nikkud_letter_entries_when_available():
    """הא / פא / צדי voweled forms must equal apply_nikkud's NO_PREFIX letter entries (PR #8)."""
    from pipeline import apply_nikkud
    entries = getattr(apply_nikkud, "NO_PREFIX_NIKKUD_DICT", None)
    if entries is None:
        pytest.skip("apply_nikkud has no NO_PREFIX_NIKKUD_DICT yet")
    assert LETTER_NAMES["ה"] == entries["הא"]
    assert LETTER_NAMES["פ"] == entries["פא"]
    assert LETTER_NAMES["צ"] == entries["צדי"]
