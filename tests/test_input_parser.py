import pytest
from pipeline.input_parser import parse_simanim_string

def test_parse_single_number():
    assert parse_simanim_string("94") == [94]
    assert parse_simanim_string(" 100 ") == [100]

def test_parse_comma_separated():
    assert parse_simanim_string("94,95,96") == [94, 95, 96]
    assert parse_simanim_string("94, 100, 102") == [94, 100, 102]

def test_parse_ranges():
    assert parse_simanim_string("1-5") == [1, 2, 3, 4, 5]
    assert parse_simanim_string("10-10") == [10]

def test_parse_complex_input():
    assert parse_simanim_string("94,95-97,100, 102-104") == [94, 95, 96, 97, 100, 102, 103, 104]

def test_parse_removes_duplicates_and_sorts():
    assert parse_simanim_string("100,99,99-101") == [99, 100, 101]

def test_parse_empty_string_raises_error():
    with pytest.raises(ValueError):
        parse_simanim_string("")
    with pytest.raises(ValueError):
        parse_simanim_string("   ")

def test_parse_invalid_chars_raises_error():
    with pytest.raises(ValueError):
        parse_simanim_string("94,95a,96")
    with pytest.raises(ValueError):
        parse_simanim_string("94;95;96")
    with pytest.raises(ValueError):
        parse_simanim_string("94+95")

def test_parse_invalid_range_raises_error():
    with pytest.raises(ValueError):
        # Missing range boundary
        parse_simanim_string("94-")
    with pytest.raises(ValueError):
        # Multiple hyphens
        parse_simanim_string("94-95-96")
    with pytest.raises(ValueError):
        # Wrong bounds order
        parse_simanim_string("100-94")

def test_parse_non_positive_numbers_raises_error():
    with pytest.raises(ValueError):
        parse_simanim_string("0")
    with pytest.raises(ValueError):
        parse_simanim_string("-5")
    with pytest.raises(ValueError):
        parse_simanim_string("10,-2,15")

def test_parse_consecutive_commas_raises_error():
    with pytest.raises(ValueError):
        parse_simanim_string("94,,95")
    with pytest.raises(ValueError):
        parse_simanim_string("94,")
