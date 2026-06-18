import pytest
from unittest.mock import patch, MagicMock
from pipeline.extractor import SefariaExtractor

@pytest.fixture
def extractor():
    return SefariaExtractor(base_url="https://fake-sefaria.org", timeout=5, retries=2)

def test_clean_html_text(extractor):
    raw_html = "<b>שלום</b> <i data-commentator=\"test\">עולם</i> &quot;ציטוט&quot;"
    cleaned = extractor.clean_html_text(raw_html)
    assert cleaned == "שלום עולם \"ציטוט\""

def test_flatten_elements_nested(extractor):
    nested_list = [
        "שורה ראשונה",
        ["שורה שנייה", ["תת-שורה שלישית"]],
        "",
        "<b>שורה רביעית</b>"
    ]
    flattened = extractor.flatten_elements(nested_list)
    assert flattened == ["שורה ראשונה", "שורה שנייה", "תת-שורה שלישית", "שורה רביעית"]

@patch("requests.get")
def test_fetch_text_v3_success(mock_get, extractor):
    # Setup mock for Sefaria v3 API success
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "versions": [
            {"language": "en", "text": "English text"},
            {"language": "he", "text": ["שורה א", "שורה ב"]}
        ]
    }
    mock_get.return_value = mock_response

    lines = extractor.fetch_text("Beit Yosef, Yoreh De'ah.94.1")
    assert lines == ["שורה א", "שורה ב"]
    # Verify that request was sent to v3 endpoint
    mock_get.assert_called_with("https://fake-sefaria.org/v3/texts/Beit_Yosef,_Yoreh_De'ah.94.1?version=hebrew", timeout=5, verify=True)

@patch("requests.get")
def test_fetch_text_v1_fallback(mock_get, extractor):
    # Setup mock so v3 fails (returns empty or throws) and v1 succeeds
    v3_response = MagicMock()
    v3_response.status_code = 404
    
    v1_response = MagicMock()
    v1_response.status_code = 200
    v1_response.json.return_value = {
        "he": ["שורה מתוצרת v1"]
    }
    
    # Mock requests.get side effect to fail on v3, succeed on v1
    # Recall that extractor tries retries * v3, then retries * v1.
    # We set retries=2. So calls 1 and 2 are v3, calls 3 and 4 are v1.
    mock_get.side_effect = [v3_response, v3_response, v1_response]

    lines = extractor.fetch_text("Tur, Yoreh De'ah.94.1")
    assert lines == ["שורה מתוצרת v1"]
    
    # Assert that it did fall back and queried v1
    assert mock_get.call_count == 3
    mock_get.assert_any_call("https://fake-sefaria.org/texts/Tur,_Yoreh_De'ah.94.1?context=0", timeout=5, verify=True)

@patch("requests.get")
def test_compile_simanim_context(mock_get, extractor):
    # Simulates success return for Sefaria works
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "versions": [
            {"language": "he", "text": ["טקסט דוגמה"]}
        ]
    }
    mock_get.return_value = mock_response

    # Compile context for Siman 94
    context = extractor.compile_simanim_context([94])
    
    assert "=== סימן 94 ===" in context
    assert "--- Tur (סימן 94) ---" in context
    assert "--- Beit Yosef (סימן 94) ---" in context
    assert "--- Shulchan Arukh (סימן 94) ---" in context
    assert "[1] טקסט דוגמה" in context


@patch("requests.get")
def test_compile_simanim_context_with_target_simanim(mock_get, extractor):
    # Simulates success return for Sefaria works
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "versions": [
            {"language": "he", "text": ["טקסט דוגמה"]}
        ]
    }
    mock_get.return_value = mock_response

    # Compile context for Siman 93 and 94, with only 94 as target
    context = extractor.compile_simanim_context([93, 94], target_simanim=[94])
    
    assert "=== סימן 93 ===" in context
    assert "--- Tur (סימן 93) ---" in context
    assert "--- Shulchan Arukh (סימן 93) ---" in context
    assert "--- Beit Yosef (סימן 93) ---" not in context
    assert "--- Shach (סימן 93) ---" not in context
    assert "--- Taz (סימן 93) ---" not in context
    
    assert "=== סימן 94 ===" in context
    assert "--- Tur (סימן 94) ---" in context
    assert "--- Beit Yosef (סימן 94) ---" in context
    assert "--- Shulchan Arukh (סימן 94) ---" in context
    assert "--- Shach (סימן 94) ---" in context
    assert "--- Taz (סימן 94) ---" in context


@patch("requests.get")
def test_fetch_text_ssl_disabled(mock_get):
    # Test that SefariaExtractor passes verify=False when ssl_verify=False
    extractor_no_ssl = SefariaExtractor(base_url="https://fake-sefaria.org", timeout=5, retries=1, ssl_verify=False)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "versions": [{"language": "he", "text": ["שורה"]}]
    }
    mock_get.return_value = mock_response

    lines = extractor_no_ssl.fetch_text("Tur, Yoreh De'ah.94.1")
    assert lines == ["שורה"]
    mock_get.assert_called_with("https://fake-sefaria.org/v3/texts/Tur,_Yoreh_De'ah.94.1?version=hebrew", timeout=5, verify=False)


def test_extractor_section_name():
    extractor_oc = SefariaExtractor(section_name="Orach Chayim", base_url="https://fake-sefaria.org", timeout=5, retries=1)
    assert extractor_oc.works["Tur"] == "Tur, Orach Chayim"
    assert extractor_oc.works["Magen Avraham"] == "Magen Avraham"
    assert extractor_oc.works["Taz"] == "Turei Zahav on Shulchan Arukh, Orach Chayim"
    assert extractor_oc.works["Mishna Berura"] == "Mishnah Berurah"
    
    with pytest.raises(ValueError):
        SefariaExtractor(section_name="Invalid Section")


