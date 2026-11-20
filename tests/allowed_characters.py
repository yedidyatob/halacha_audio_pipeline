"""The repo config.yaml text_check.allowed_characters, for tests whose stand-in config needs it."""
import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    ALLOWED_CHARACTERS = yaml.safe_load(_f)["text_check"]["allowed_characters"]
