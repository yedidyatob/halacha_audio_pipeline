"""The text check: characters outside config text_check.allowed_characters (e.g. a stray Arabic or Cyrillic letter)."""
import re
from typing import Optional

from pipeline.generator import BatchResultError

SHOWN_CHARACTERS = 5  # bad characters quoted with their context in a report; the rest are counted
CONTEXT_CHARS = 15    # characters of context on each side of a quoted bad character


class DisallowedCharactersError(BatchResultError):
    """
    A model response contains characters outside text_check.allowed_characters. Like a blocked or empty
    response, a new call may fix it: sync mode retries, batch mode resubmits (generator.max_retries).
    """


def disallowed_characters_report(text: str, allowed_characters: str) -> Optional[str]:
    """None if every character of text is allowed; otherwise a report naming each bad character with context."""
    bad = [(m.group(), m.start()) for m in re.finditer(f"[^{allowed_characters}]", text)]
    if not bad:
        return None
    shown = "; ".join(f"'{c}' U+{ord(c):04X} in '...{text[max(0, i - CONTEXT_CHARS):i + CONTEXT_CHARS + 1]}...'"
                      for c, i in bad[:SHOWN_CHARACTERS])
    more = f" (+{len(bad) - SHOWN_CHARACTERS} more)" if len(bad) > SHOWN_CHARACTERS else ""
    return f"{len(bad)} character(s) outside text_check.allowed_characters: {shown}{more}"


def check_allowed_characters(text: str, allowed_characters: str, label: str) -> str:
    """Returns text unchanged, or raises DisallowedCharactersError with the report."""
    report = disallowed_characters_report(text, allowed_characters)
    if report:
        raise DisallowedCharactersError(f"{label}: {report}")
    return text
