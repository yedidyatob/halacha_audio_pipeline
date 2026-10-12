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
