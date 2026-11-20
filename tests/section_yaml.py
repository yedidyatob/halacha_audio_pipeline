"""The repo config.yaml 'sections' block, for tests that build their own config YAML."""
import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    SECTIONS = yaml.safe_load(_f)["sections"]

SECTIONS_YAML = yaml.safe_dump({"sections": SECTIONS}, allow_unicode=True, sort_keys=False)
