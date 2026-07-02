import pytest
from unittest.mock import patch, MagicMock
from pipeline.evaluator import evaluate_draft, get_seif_from_ref, get_clean_gematria_without_quotes

def test_gematria_helpers():
    assert get_clean_gematria_without_quotes(1) == "א"
    assert get_clean_gematria_without_quotes(15) == "טו"
    assert get_clean_gematria_without_quotes(94) == "צד"
    
    assert get_seif_from_ref("Shulchan Arukh, Yoreh De'ah 94:3", 94) == 3
    assert get_seif_from_ref("Shulchan Arukh, Yoreh De'ah 95:1", 94) is None

@patch("requests.get")
@patch("pipeline.extractor.SefariaExtractor.fetch_siman_sources")
def test_evaluate_draft_success(mock_fetch_sources, mock_get):
    # Mock Shulchan Arukh sources for Siman 94 (3 Se'ifim, Se'if 2 has Rema gloss "הגה")
    mock_fetch_sources.return_value = {
        "Shulchan Arukh": [
            "סעיף א: דין כלשהו",
            "סעיף ב: הגה דין נוסף",
            "סעיף ג: דין שלישי"
        ],
        "Tur": ["טקסט טור"],
        "Beit Yosef": ["טקסט בית יוסף"],
        "Shach": ["טקסט שך"],
        "Taz": ["טקסט טז"]
    }
    
    # Mock links for Shach on Se'if 1 and Taz on Se'if 2
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "ref": "Siftei Kohen on Shulchan Arukh, Yoreh De'ah 94:1:1",
            "anchorRef": "Shulchan Arukh, Yoreh De'ah 94:1"
        },
        {
            "ref": "Turei Zahav on Shulchan Arukh, Yoreh De'ah 94:2:1",
            "anchorRef": "Shulchan Arukh, Yoreh De'ah 94:2"
        }
    ]
    mock_get.return_value = mock_response

    # Prepare draft text that satisfies all conditions:
    # Se'if 1 (א): Mechaber, Shach
    # Se'if 2 (ב): Mechaber, Rema, Taz
    # Se'if 3 (ג): Mechaber
    # Global: Tur, Beit Yosef
    draft_text = """
# סימן צד
הטור והבית יוסף דנים בדינים האלה.

## סעיף א
המחבר אומר שזה מותר, והשפתי כהן (הש"ך) מסכים.

## סעיף ב
מרן השולחן ערוך מביא דעה ראשונה, והרמ"א בהגה אומר שנוהגים לאסור, והטורי זהב (הט"ז) דן בדבריהם.

## סעיף ג
המחבר פוסק להלכה.
"""

    result = evaluate_draft(siman=94, draft_text=draft_text, config_path="config.yaml")
    
    assert result["success"] is True
    assert result["total_localized_flags"] == 0
    assert result["total_general_flags"] == 0
    assert "Status: EXCELLENT" in result["report"]

@patch("requests.get")
@patch("pipeline.extractor.SefariaExtractor.fetch_siman_sources")
def test_evaluate_draft_fallback_success(mock_fetch_sources, mock_get):
    # Mock Shulchan Arukh sources for Siman 94 (3 Se'ifim, Se'if 2 has Rema gloss "הגה")
    mock_fetch_sources.return_value = {
        "Shulchan Arukh": [
            "סעיף א: דין כלשהו",
            "סעיף ב: הגה דין נוסף",
            "סעיף ג: דין שלישי"
        ],
        "Tur": ["טקסט טור"],
        "Beit Yosef": ["טקסט בית יוסף"],
        "Shach": ["טקסט שך"],
        "Taz": ["טקסט טז"]
    }
    
    # Mock links for Shach on Se'if 1 and Taz on Se'if 2
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "ref": "Siftei Kohen on Shulchan Arukh, Yoreh De'ah 94:1:1",
            "anchorRef": "Shulchan Arukh, Yoreh De'ah 94:1"
        },
        {
            "ref": "Turei Zahav on Shulchan Arukh, Yoreh De'ah 94:2:1",
            "anchorRef": "Shulchan Arukh, Yoreh De'ah 94:2"
        }
    ]
    mock_get.return_value = mock_response

    # Draft text using the bold fallback format and asterisks
    draft_text = """
להלן מיפוי היררכי ומלא של סימן צ"ד.
הטור והבית יוסף דנים בדינים האלה.

**הסעיף בשולחן ערוך:** מקרה א'
המחבר אומר שזה מותר, והשפתי כהן (הש"ך) מסכים.
***

**הסעיף בשולחן ערוך:** מקרה ב'
מרן השולחן ערוך מביא דעה ראשונה, והרמ"א בהגה אומר שנוהגים לאסור, והטורי זהב (הט"ז) דן בדבריהם.
***

**הסעיף בשולחן ערוך:** מקרה ג'
המחבר פוסק להלכה.
"""

    result = evaluate_draft(siman=94, draft_text=draft_text, config_path="config.yaml")
    
    assert result["success"] is True
    assert result["total_localized_flags"] == 0
    assert result["total_general_flags"] == 0
    assert "Status: EXCELLENT" in result["report"]

@patch("requests.get")
@patch("pipeline.extractor.SefariaExtractor.fetch_siman_sources")
def test_evaluate_draft_failures(mock_fetch_sources, mock_get):
    # Mock Shulchan Arukh sources for Siman 94 (2 Se'ifim)
    mock_fetch_sources.return_value = {
        "Shulchan Arukh": [
            "סעיף א: דין כלשהו",
            "סעיף ב: הגה דין נוסף"
        ]
    }
    
    # Mock links: Shach on Se'if 1
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "ref": "Siftei Kohen on Shulchan Arukh, Yoreh De'ah 94:1:1",
            "anchorRef": "Shulchan Arukh, Yoreh De'ah 94:1"
        }
    ]
    mock_get.return_value = mock_response

    # Draft has:
    # - Missing Tur / Beit Yosef globally
    # - Se'if 1 draft block is missing Shach
    # - Se'if 2 draft block is missing entirely
    draft_text = """
## סעיף א
המחבר פוסק להלכה.
"""

    result = evaluate_draft(siman=94, draft_text=draft_text, config_path="config.yaml")
    
    assert result["success"] is False
    assert result["total_localized_flags"] >= 2  # Missing Shach in Se'if 1 + Se'if 2 missing block
    assert result["total_general_flags"] == 2    # Missing Tur & Beit Yosef globally
    assert 2 in result["missing_seifim"]
    assert "Status: WARNING" in result["report"]
