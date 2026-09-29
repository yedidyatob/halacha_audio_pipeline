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
        assert "×©\"×š ×•×˜\"×–" in config.gemini_system_instruction
        assert "×©×¤×ª×™ ×›×”×Ÿ" in config.gemini_system_instruction
        assert "×”×©×¤×ª×™ ×›×”×Ÿ" in config.relations_instruction
        assert "×™×•×¨×” ×“×¢×”" in config.relations_instruction
        assert "×ž×”×¨×©\"×œ" in config.polishing_instruction

    # Test Orach Chayim
    with patch("builtins.open", mock_open(read_data=yaml_content.format(section="Orach Chayim"))), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        config = PipelineConfig("config.yaml")
        assert "×ž×’×Ÿ ××‘×¨×”×, ×˜×•×¨×™ ×–×”×‘" in config.gemini_system_instruction
        assert "×”×ž×’×Ÿ ××‘×¨×”×" in config.gemini_system_instruction
        assert "×”×ž×’×Ÿ ××‘×¨×”×" in config.relations_instruction
        assert "××•×¨×— ×—×™×™×" in config.relations_instruction
        assert "×ž×©× ×” ×‘×¨×•×¨×”" in config.polishing_instruction



def test_pipeline_config_lesson_framing_optional():
    yaml_content = """
halachic_section: "Yoreh De'ah"
generator:
  engine: "gemini"
"""
    with patch("builtins.open", mock_open(read_data=yaml_content)), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        config = PipelineConfig("config.yaml")
        assert config.lesson_intro_template == ""
        assert config.lesson_outro_template == ""


def test_pipeline_config_lesson_framing_loaded():
    yaml_content = """
halachic_section: "Yoreh De'ah"
generator:
  engine: "gemini"
lesson_framing:
  intro: "×©×œ×•×, ×¡×™×ž×Ÿ {spoken_siman}."
  outro: "×œ×”×ª×¨××•×ª, ×¡×™×ž×Ÿ {gematria_siman}."
"""
    with patch("builtins.open", mock_open(read_data=yaml_content)), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        config = PipelineConfig("config.yaml")
        assert "spoken_siman" in config.lesson_intro_template
        assert "gematria_siman" in config.lesson_outro_template
