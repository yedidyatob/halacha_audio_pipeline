import pytest
import os
from unittest.mock import patch, mock_open
from pipeline.config import PipelineConfig

def test_pipeline_config_ssl_verify_default():
    # Mock yaml data with ssl_verify missing
    yaml_content = """
halachic_section: "Yoreh De'ah"
api_keys:
  gemini_api_key: "fake-gemini"
sefaria:
  base_url: "https://www.sefaria.org/api"
generator:
  engine: "gemini"
"""
    with patch("builtins.open", mock_open(read_data=yaml_content)), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        config = PipelineConfig("config.yaml")
        assert config.ssl_verify is True

def test_pipeline_config_ssl_verify_false():
    # Mock yaml data with ssl_verify set to false
    yaml_content = """
halachic_section: "Yoreh De'ah"
api_keys:
  gemini_api_key: "fake-gemini"
sefaria:
  base_url: "https://www.sefaria.org/api"
  ssl_verify: false
generator:
  engine: "gemini"
"""
    with patch("builtins.open", mock_open(read_data=yaml_content)), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        config = PipelineConfig("config.yaml")
        assert config.ssl_verify is False


def test_pipeline_config_prompt_formatting():
    # Mock yaml data with the instruction templates
    yaml_content = """
halachic_section: "{section}"
system_instruction: "Including {{commentators_list_hebrew}} and {{prompt_commentators_desc}}"
polishing_instruction: "Including {{commentators_list_short}} and {{tts_abbreviations}}"
relations_instruction: "In {{hebrew_name}} with {{prompt_commentators_desc}}"
generator:
  engine: "gemini"
"""

    # Test Yoreh De'ah
    with patch("builtins.open", mock_open(read_data=yaml_content.format(section="Yoreh De'ah"))), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        config = PipelineConfig("config.yaml")
        assert "ש\"ך וט\"ז" in config.gemini_system_instruction
        assert "שפתי כהן" in config.gemini_system_instruction
        assert "השפתי כהן" in config.relations_instruction
        assert "יורה דעה" in config.relations_instruction
        assert "מהרש\"ל" in config.polishing_instruction

    # Test Orach Chayim
    with patch("builtins.open", mock_open(read_data=yaml_content.format(section="Orach Chayim"))), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        config = PipelineConfig("config.yaml")
        assert "מגן אברהם, טורי זהב" in config.gemini_system_instruction
        assert "המגן אברהם" in config.gemini_system_instruction
        assert "המגן אברהם" in config.relations_instruction
        assert "אורח חיים" in config.relations_instruction
        assert "משנה ברורה" in config.polishing_instruction

