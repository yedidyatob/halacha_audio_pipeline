"""The repo config.yaml text_check settings, for tests whose stand-in config needs them."""
import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    _TEXT_CHECK = yaml.safe_load(_f)["text_check"]
ALLOWED_CHARACTERS = _TEXT_CHECK["allowed_characters"]
MAX_SENTENCE_WORDS = _TEXT_CHECK["max_sentence_words"]
