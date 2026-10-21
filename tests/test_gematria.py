import pytest
from pipeline.gematria import int_to_gematria

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
