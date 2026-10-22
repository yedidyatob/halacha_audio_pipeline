import pytest
import os
from unittest.mock import patch, mock_open
from pipeline.config import PipelineConfig

REQUIRED_YAML = """
halachic_section: "Yoreh De'ah"
sefaria:
  base_url: "https://www.sefaria.org/api"
  timeout_seconds: 15
  retries: 3
  ssl_verify: false
directories:
  thinking_dir: "./output/thinking"
generator:
  engine: "gemini"
tts:
  engine: "elevenlabs"
  chunk_gap_paragraph_ms: 700
  chunk_gap_sentence_ms: 300
  elevenlabs:
    chunk_size: 8000
    context_chars: 500
"""


def _load(yaml_content):
    with patch("builtins.open", mock_open(read_data=yaml_content)), \
         patch("os.path.exists", return_value=True), \
         patch("os.makedirs"):
        return PipelineConfig("config.yaml")


def test_sefaria_and_engine_settings_are_read():
    config = _load(REQUIRED_YAML)
    assert (config.sefaria_base_url, config.sefaria_timeout, config.sefaria_retries, config.ssl_verify) == \
        ("https://www.sefaria.org/api", 15, 3, False)
    assert (config.generator_engine, config.tts_engine) == ("gemini", "elevenlabs")
    assert _load(REQUIRED_YAML.replace("ssl_verify: false", "ssl_verify: true")).ssl_verify is True


@pytest.mark.parametrize("line, message", [
    ('  base_url: "https://www.sefaria.org/api"\n', "'base_url' under 'sefaria'"),
    ("  timeout_seconds: 15\n", "'timeout_seconds' under 'sefaria'"),
    ("  retries: 3\n", "'retries' under 'sefaria'"),
    ("  ssl_verify: false\n", "'ssl_verify' under 'sefaria'"),
    ('generator:\n  engine: "gemini"\n', "'engine' under 'generator'"),
    ('  engine: "elevenlabs"\n', "'engine' under 'tts'"),
])
def test_sefaria_and_engine_settings_are_required(line, message):
    assert REQUIRED_YAML.count(line) == 1
    with pytest.raises(ValueError, match=message):
        _load(REQUIRED_YAML.replace(line, ""))


@pytest.mark.parametrize("old, new, message", [
    ('base_url: "https://www.sefaria.org/api"', 'base_url: ""', "sefaria.base_url' must be a non-empty string"),
    ("timeout_seconds: 15", "timeout_seconds: 0", "sefaria.timeout_seconds' must be a positive number"),
    ("timeout_seconds: 15", 'timeout_seconds: "15"', "sefaria.timeout_seconds' must be a positive number"),
    ("retries: 3", "retries: 0", "sefaria.retries' must be a positive integer"),
    ("retries: 3", "retries: 1.5", "sefaria.retries' must be a positive integer"),
    ("ssl_verify: false", 'ssl_verify: "no"', "sefaria.ssl_verify' must be true or false"),
    ('engine: "gemini"', 'engine: "claude"', "Invalid generator.engine choice: 'claude'"),
    ('engine: "elevenlabs"', 'engine: "azure"', "Invalid tts.engine choice: 'azure'"),
])
def test_sefaria_and_engine_settings_are_validated(old, new, message):
    with pytest.raises(ValueError, match=message):
        _load(REQUIRED_YAML.replace(old, new, 1))


def test_pipeline_config_prompt_formatting():
    # Mock yaml data with the instruction templates
    yaml_content = """
halachic_section: "{section}"
sefaria:
  base_url: "https://www.sefaria.org/api"
  timeout_seconds: 15
  retries: 3
  ssl_verify: true
directories:
  thinking_dir: "./output/thinking"
system_instruction: "Including {{commentators_list_hebrew}} and {{prompt_commentators_desc}}"
polishing_instruction: "Including {{commentators_list_short}} and {{tts_abbreviations}}"
relations_instruction: "In {{hebrew_name}} with {{prompt_commentators_desc}}"
pronunciation:
  always_vocalize: []
  never_vocalize: []
generator:
  engine: "gemini"
tts:
  engine: "elevenlabs"
  chunk_gap_paragraph_ms: 700
  chunk_gap_sentence_ms: 300
  elevenlabs:
    chunk_size: 8000
    context_chars: 500
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

