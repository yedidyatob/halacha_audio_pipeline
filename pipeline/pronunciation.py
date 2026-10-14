"""
Pronunciation stage: wraps words the TTS engine is likely to mispronounce in
ElevenLabs-style SSML phoneme tags with IPA:

    <phoneme alphabet="ipa" ph="tet">טית</phoneme>

The tags are passed unchanged to every TTS engine (ElevenLabs v3/v4 honors them;
Google TTS neither applies nor reads them aloud), so there is no per-engine logic here.

Safety: the LLM must return the script word for word. The output is validated by
stripping the tags and comparing with the input; on any difference, or on a malformed
tag, the ORIGINAL untagged script is returned and a warning is logged.
"""

from __future__ import annotations

import difflib
import re
from typing import Callable, Optional

from pipeline.logger import get_logger

logger = get_logger(__name__)

# A single, well-formed tag: exact attribute form, non-empty ph, non-empty inner word,
# no nested tags / angle brackets and no line breaks inside.
PHONEME_TAG_RE = re.compile(
    r'<phoneme alphabet="ipa" ph="(?P<ph>[^"<>\n\r]+)">(?P<word>[^<>\n\r]+)</phoneme>'
)

# Anything that looks like a phoneme open or close tag (used to detect malformed markup).
_ANY_PHONEME_MARKUP_RE = re.compile(r"</?\s*phoneme\b[^>]*>?", re.IGNORECASE)

# Lenient span pattern used by consumers (nikkud, TTS chunking) to protect tag spans.
PHONEME_SPAN_RE = re.compile(r"<phoneme\b[^>]*>.*?</phoneme>", re.IGNORECASE | re.DOTALL)


def find_phoneme_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of every <phoneme ...>...</phoneme> span in text."""
    return [m.span() for m in PHONEME_SPAN_RE.finditer(text)]


def split_phoneme_segments(text: str) -> list[tuple[str, bool]]:
    """Splits text into [(segment, is_tag), ...] preserving order; concatenation == text."""
    segments: list[tuple[str, bool]] = []
    pos = 0
    for start, end in find_phoneme_spans(text):
        if start > pos:
            segments.append((text[pos:start], False))
        segments.append((text[start:end], True))
        pos = end
    if pos < len(text) or not segments:
        segments.append((text[pos:], False))
    return segments


def strip_phoneme_tags(text: str) -> str:
    """Removes well-formed phoneme tags, keeping the inner (original) word."""
    return PHONEME_TAG_RE.sub(lambda m: m.group("word"), text)


def extract_phoneme_pairs(text: str) -> list[tuple[str, str]]:
    """List of (word, ipa) for every well-formed phoneme tag, in order of appearance."""
    return [(m.group("word"), m.group("ph")) for m in PHONEME_TAG_RE.finditer(text)]


def find_malformed_tags(text: str) -> list[str]:
    """
    Returns descriptions of malformed phoneme markup (empty list = all good):
    unclosed / stray / nested tags, missing or empty ph, wrong alphabet, line breaks inside.
    """
    problems = []
    remainder = PHONEME_TAG_RE.sub("", text)  # remove every valid tag
    for m in _ANY_PHONEME_MARKUP_RE.finditer(remainder):
        problems.append(m.group(0)[:80])
    return problems


def _normalize(text: str) -> str:
    """Only trivial differences: line endings, trailing whitespace per line and at the end."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def _diff_summary(expected: str, actual: str, max_lines: int = 6) -> str:
    diff = list(difflib.unified_diff(
        expected.split("\n"), actual.split("\n"), "input", "output(stripped)", n=0, lineterm=""
    ))
    changed = [l for l in diff if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    shown = [l[:120] for l in changed[:max_lines]]
    more = f" ... (+{len(changed) - max_lines} more)" if len(changed) > max_lines else ""
    return f"{len(changed)} changed line(s): " + " | ".join(shown) + more


def validate_tagged_output(original: str, tagged: str) -> Optional[str]:
    """Returns None if `tagged` is a valid tagging of `original`, else a reason string."""
    if not tagged or not tagged.strip():
        return "LLM returned empty output"
    malformed = find_malformed_tags(tagged)
    if malformed:
        return f"malformed phoneme tag(s): {malformed[:5]}"
    expected = _normalize(original)
    actual = _normalize(strip_phoneme_tags(tagged))
    if expected != actual:
        return "text changed outside tags: " + _diff_summary(expected, actual)
    return None


def tag_pronunciation(script_text: str, generate: Callable[[str], str]) -> tuple[str, list[tuple[str, str]]]:
    """
    Sends the script to the LLM via `generate(script_text) -> tagged_text` and validates.

    Returns (text, pairs). On any validation failure (or LLM error) logs a warning and
    returns (script_text, []) - the ORIGINAL untagged script.
    """
    try:
        tagged = generate(script_text)
    except Exception as e:
        logger.warning(f"Pronunciation tagging LLM call failed: {e}. Using the untagged script.")
        return script_text, []

    reason = validate_tagged_output(script_text, tagged)
    if reason:
        logger.warning(f"Pronunciation tagging rejected ({reason}). Using the untagged script.")
        return script_text, []

    tagged = tagged.replace("\r\n", "\n")
    pairs = extract_phoneme_pairs(tagged)
    logger.info(f"Pronunciation tagging: {len(pairs)} word(s) tagged.")
    return tagged, pairs


def format_pairs(pairs: list[tuple[str, str]]) -> str:
    """'word<TAB>ipa' per line."""
    return "".join(f"{word}\t{ipa}\n" for word, ipa in pairs)


def run_pronunciation_stage(script_text: str, config, generator_factory=None):
    """
    Pipeline entry point. Returns (text, pairs) or None when pronunciation.enabled is false
    (the stage is skipped entirely and no LLM client is created).
    """
    if not config.pronunciation_enabled:
        logger.info("Pronunciation tagging disabled (pronunciation.enabled: false); skipping.")
        return None
    if generator_factory is None:
        from pipeline.factory import create_pronunciation_generator as generator_factory
    generator = generator_factory(config)
    instruction = config.pronunciation_instruction
    logger.info(f"Running pronunciation tagging via {config.pronunciation_engine} ({generator.model_name})...")
    return tag_pronunciation(script_text, lambda text: generator.generate_text(instruction, text))
