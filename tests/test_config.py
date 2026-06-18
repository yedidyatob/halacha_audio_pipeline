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
