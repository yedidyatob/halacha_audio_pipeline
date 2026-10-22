import pytest
from unittest.mock import patch, MagicMock
from pipeline.extractor import SefariaExtractor

@pytest.fixture
def extractor():
    return SefariaExtractor(section_name="Yoreh De'ah", base_url="https://fake-sefaria.org", timeout=5, retries=2, ssl_verify=True)

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
    extractor_no_ssl = SefariaExtractor(section_name="Yoreh De'ah", base_url="https://fake-sefaria.org", timeout=5, retries=1, ssl_verify=False)
    
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
    extractor_oc = SefariaExtractor(section_name="Orach Chayim", base_url="https://fake-sefaria.org", timeout=5, retries=1, ssl_verify=True)
    assert extractor_oc.works["Tur"] == "Tur, Orach Chayim"
    assert extractor_oc.works["Magen Avraham"] == "Magen Avraham"
    assert extractor_oc.works["Taz"] == "Turei Zahav on Shulchan Arukh, Orach Chayim"
    assert extractor_oc.works["Mishna Berura"] == "Mishnah Berurah"
    
    with pytest.raises(ValueError):
        SefariaExtractor(section_name="Invalid Section", base_url="https://fake-sefaria.org", timeout=5, retries=1, ssl_verify=True)




# --- Rama labels (Shulchan Arukh) -------------------------------------------------
# Raw se'if HTML from Sefaria; the Rama's glosses are in <small>.

import json
import os
import re

from pipeline.extractor import RAMA_OPEN, RAMA_CLOSE

with open(os.path.join(os.path.dirname(__file__), "fixtures", "shulchan_arukh_yd_rama.json"), encoding="utf-8") as f:
    RAW_SEIFIM = json.load(f)["seifim"]


def _labeled(extractor, seif):
    return extractor.flatten_elements(RAW_SEIFIM[seif], label_rama=True)[0]


def _split(text):
    """(mechaber_segments, rama_segments) of a labeled se'if."""
    pattern = re.escape(RAMA_OPEN) + r" (.*?) " + re.escape(RAMA_CLOSE)
    return [p.strip() for p in re.split(pattern, text)[0::2]], re.findall(pattern, text)


def test_inline_rama_gloss_is_labeled(extractor):
    text = _labeled(extractor, "89:3")
    mechaber, rama = _split(text)
    assert rama[0] == '(ויש מצריכים נטילה) (שערים והגהות ש"ד)'  # inline gloss, no הגה
    assert rama[1].startswith("ושומן של בשר דינו כבשר עצמו")      # leading הגה dropped
    assert mechaber[1].startswith("אבל אם בא לאכול הגבינה עצמה")
    assert "<" not in text and "הגה ושומן" not in text


def test_mechaber_after_gloss_is_unlabeled(extractor):
    mechaber, rama = _split(_labeled(extractor, "91:5"))
    assert rama[0].startswith('וי"א דלאחר ששהה במלחו')
    assert rama[0].endswith("וטוב להחמיר במקו' שאין הפסד מרובה:")
    assert mechaber[1].startswith("ומליח כעין מליח דבעי לה לארחא")


def test_several_glosses_in_one_seif(extractor):
    mechaber, rama = _split(_labeled(extractor, "92:2"))
    assert rama[0] == "(ולדידן דאין סומכין אעובדי כוכבים בעינן ששים בחתיכה ואם לאו כולה אסורה)"
    assert mechaber[1].startswith("ומשערין בכולה")
    assert rama[1].startswith("וכן אם לא ניער כלל")
    assert any(m.startswith("וכן אם נפל לתוך המרק") for m in mechaber[2:])


def test_seif_with_only_inline_glosses(extractor):
    text = _labeled(extractor, "94:1")
    mechaber, rama = _split(text)
    assert "הגה" not in text.split()  # no הגה word in the raw se'if either
    assert rama == ['(אם הכף בן יומו דהיינו ששמשו בו בכלי ראשון תוך מעת לעת) (ארוך כלל ל"ז)',
                    '(וסברא ראשונה עיקר וכן נוהגין) (מרדכי וש"ד והגהותיו וארוך) (וע"ל סי\' צ"ח):']
    assert mechaber[1] == "ויש מי שאומר שאם הכף של מתכת משערים בכולו משום דחם מקצתו חם כולו"


def test_unmarked_parenthetical_stays_unlabeled(extractor):
    # 96:3: Sefaria did not put this gloss in <small>; the prompt covers it
    text = _labeled(extractor, "96:3")
    assert RAMA_OPEN not in text and '(וי"א אפי\' אינו בן יומו)' in text


def test_nested_and_unclosed_small(extractor):
    assert extractor.label_rama_glosses("א <small>ב <small>(טור)</small>.</small> ג") == \
        f"א  {RAMA_OPEN} ב (טור). {RAMA_CLOSE}  ג"
    assert extractor.clean_html_text(extractor.label_rama_glosses("א <small>הגה: ב")) == f"א {RAMA_OPEN} ב {RAMA_CLOSE}"
    assert extractor.label_rama_glosses("א <small> </small>ב") == "א  ב"


def test_flatten_without_label_rama_keeps_plain_cleaning(extractor):
    assert extractor.flatten_elements(["א <small>הגה ב</small>"]) == ["א הגה ב"]


def _sefaria_by_ref(url, **kwargs):
    response = MagicMock(status_code=200)
    response.json.return_value = {"versions": [{"language": "he", "text": ["מחבר <small>הגה רמא</small> סוף"]}]}
    return response


@patch("requests.get", side_effect=_sefaria_by_ref)
def test_only_shulchan_arukh_is_labeled(mock_get, extractor):
    sources = extractor.fetch_siman_sources(94)
    assert sources["Shulchan Arukh"] == [f"מחבר {RAMA_OPEN} רמא {RAMA_CLOSE} סוף"]
    for work in ("Tur", "Beit Yosef", "Shach", "Taz"):
        assert sources[work] == ["מחבר הגה רמא סוף"]
    context = extractor.compile_simanim_context([94])
    assert f"[1] מחבר {RAMA_OPEN} רמא {RAMA_CLOSE} סוף" in context
    assert context.count(RAMA_OPEN) == 1


def test_stage1_prompt_describes_the_rama_labels():
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        instruction = PipelineConfig("config.yaml").gemini_system_instruction
    assert f"{RAMA_OPEN} ... {RAMA_CLOSE}" in instruction
