"""
Pronunciation stage: wraps words the TTS engine is likely to mispronounce in
ElevenLabs-style SSML phoneme tags with IPA:

    <phoneme alphabet="ipa" ph="tet">טית</phoneme>

The tags are passed unchanged to every TTS engine (ElevenLabs v3/v4 honors them;
Google TTS neither applies nor reads them aloud), so there is no per-engine logic here.

Safety: the LLM must return the script word for word. The output is validated by
stripping the tags and comparing with the input; on any difference, a malformed tag, or
an LLM call failure, PronunciationTaggingError is raised. There is no untagged fallback:
the caller decides what to do (the pipeline fails that siman).
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


class PronunciationTaggingError(RuntimeError):
    """Raised when pronunciation tagging fails (LLM error or invalid tagged output)."""


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


def parse_tagged_response(script_text: str, tagged: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Validates an LLM tagging response against the original script.
    Returns (tagged_text, pairs); raises PronunciationTaggingError if the output is invalid
    (empty, malformed tags, or text changed outside tags). Shared by sync and batch mode.
    """
    reason = validate_tagged_output(script_text, tagged)
    if reason:
        raise PronunciationTaggingError(f"pronunciation tagging output rejected: {reason}")
    tagged = tagged.replace("\r\n", "\n")
    pairs = extract_phoneme_pairs(tagged)
    logger.info(f"Pronunciation tagging: {len(pairs)} word(s) tagged.")
    return tagged, pairs


def tag_pronunciation(script_text: str, generate: Callable[[str], str]) -> tuple[str, list[tuple[str, str]]]:
    """
    Sends the script to the LLM via `generate(script_text) -> tagged_text` and validates.

    Returns (tagged_text, pairs). Raises PronunciationTaggingError if the LLM call fails
    or the output is invalid (text changed outside tags, malformed tags, empty output).
    """
    try:
        tagged = generate(script_text)
    except Exception as e:
        raise PronunciationTaggingError(f"pronunciation tagging LLM call failed: {e}") from e
    return parse_tagged_response(script_text, tagged)


def format_pairs(pairs: list[tuple[str, str]]) -> str:
    """'word<TAB>ipa' per line."""
    return "".join(f"{word}\t{ipa}\n" for word, ipa in pairs)


class PronunciationTagger:
    """
    The pronunciation-tagging stage, bound to its own generator (model, temperature),
    instruction and thinking_level from config.yaml. The same request and the same
    validation are used for a synchronous call and for a batch job.
    """

    def __init__(self, generator, instruction: str, thinking_level: Optional[str]):
        self.generator = generator
        self.instruction = instruction
        self.thinking_level = thinking_level

    @property
    def supports_batch(self) -> bool:
        from pipeline.generator import BatchCapableGenerator
        return isinstance(self.generator, BatchCapableGenerator)

    def tag(self, script_text: str) -> tuple[str, list[tuple[str, str]]]:
        """Synchronous tagging. Raises PronunciationTaggingError on any failure."""
        return tag_pronunciation(
            script_text,
            lambda text: self.generator.generate_text(self.instruction, text, thinking_level=self.thinking_level),
        )

    def submit_batch(self, script_text: str, custom_id: str, cache_dir: str) -> str:
        """Submits the tagging request as a batch job; returns the job id."""
        return self.generator.submit_batch(
            system_instruction=self.instruction,
            user_prompt=script_text,
            temperature=self.generator.temperature,
            custom_id=custom_id,
            cache_dir=cache_dir,
            thinking_level=self.thinking_level,
        )

    def batch_result(self, script_text: str, job_id: str) -> tuple[str, list[tuple[str, str]]]:
        """Fetches and validates a completed batch job. Raises PronunciationTaggingError on failure."""
        try:
            tagged = self.generator.get_batch_result(job_id)
        except Exception as e:
            raise PronunciationTaggingError(f"pronunciation tagging batch result unavailable: {e}") from e
        return parse_tagged_response(script_text, tagged)


def create_pronunciation_tagger(config, generator_factory=None) -> PronunciationTagger:
    """Builds the tagger from the 'pronunciation' section of config.yaml (ignores pronunciation.enabled)."""
    if generator_factory is None:
        from pipeline.factory import create_pronunciation_generator as generator_factory
    generator = generator_factory(config)
    logger.info(f"Pronunciation tagging via {config.pronunciation_engine} ({generator.model_name}).")
    return PronunciationTagger(generator, config.pronunciation_instruction, config.pronunciation_thinking_level)
