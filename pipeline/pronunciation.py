"""
Pronunciation stage: an LLM adds nikkud ONLY to words the TTS engine could read in more
than one way (context-dependent homographs such as עובר / פת / סופר, and letter names used
as letters). The output is plain Hebrew text; there is no markup.

It runs after the dictionary nikkud stage (apply_nikkud), so its input may already contain
vocalized words; the LLM must leave those exactly as they are.

Validation (strict):
  - with all nikkud removed, the output must equal the input (only trailing whitespace and
    line endings may differ), and
  - every word that already had nikkud in the input must be unchanged in the output.
On any violation, or an LLM call failure, PronunciationError is raised. There is no
fallback: the caller decides what to do (the pipeline fails that siman).
"""

from __future__ import annotations

import difflib
import re
from typing import Callable, Optional

from pipeline.logger import get_logger

logger = get_logger(__name__)

# Hebrew points and cantillation marks. Excludes maqaf (U+05BE), paseq (U+05C0),
# sof pasuq (U+05C3) and nun hafukha (U+05C6), which are punctuation, not nikkud.
_NIKKUD_RE = re.compile("[\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]")

# Markup the LLM must never emit (e.g. leftover SSML / XML / HTML tags).
_MARKUP_RE = re.compile(r"</?[A-Za-z][^<>]*>")

# Punctuation trimmed from the ends of a word in the review list of changed words.
_EDGE_PUNCT = ".,;:!?()[]{}-\u05BE\u2013\u2014"

Change = tuple[str, str]  # (original word, vocalized word)


class PronunciationError(RuntimeError):
    """Raised when the pronunciation stage fails (LLM error or invalid output)."""


def strip_nikkud(text: str) -> str:
    """Removes Hebrew nikkud and cantillation marks, keeping letters and punctuation."""
    return _NIKKUD_RE.sub("", text)


def has_nikkud(text: str) -> bool:
    return bool(_NIKKUD_RE.search(text))


def find_markup(text: str) -> list[str]:
    """Tag-like markup (<...>) found in text."""
    return [m.group(0)[:80] for m in _MARKUP_RE.finditer(text)]


def _normalize(text: str) -> str:
    """Only trivial differences: line endings, trailing whitespace per line and at the end."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


def _diff_summary(expected: str, actual: str, max_lines: int = 6) -> str:
    diff = list(difflib.unified_diff(
        expected.split("\n"), actual.split("\n"), "input", "output", n=0, lineterm=""
    ))
    changed = [l for l in diff if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    shown = [l[:120] for l in changed[:max_lines]]
    more = f" ... (+{len(changed) - max_lines} more)" if len(changed) > max_lines else ""
    return f"{len(changed)} changed line(s): " + " | ".join(shown) + more


def _word_pairs(original: str, vocalized: str) -> list[Change]:
    """
    Aligned (input word, output word) pairs. Only valid once both texts are known to be
    equal without nikkud, so whitespace splitting yields the same number of words.
    """
    return list(zip(_normalize(original).split(), _normalize(vocalized).split()))


def validate_vocalized_output(original: str, vocalized: str) -> Optional[str]:
    """Returns None if `vocalized` is a valid output for `original`, else a reason string."""
    if not vocalized or not vocalized.strip():
        return "LLM returned empty output"
    markup = [m for m in find_markup(vocalized) if m not in original]
    if markup:
        return f"markup in output: {markup[:5]}"
    expected = strip_nikkud(_normalize(original))
    actual = strip_nikkud(_normalize(vocalized))
    if expected != actual:
        return "text changed (ignoring nikkud): " + _diff_summary(expected, actual)
    clobbered = [(a, b) for a, b in _word_pairs(original, vocalized) if a != b and has_nikkud(a)]
    if clobbered:
        shown = ", ".join(f"{a} -> {b}" for a, b in clobbered[:5])
        return f"{len(clobbered)} already-vocalized word(s) changed: {shown}"
    return None


def changed_words(original: str, vocalized: str) -> list[Change]:
    """(original, vocalized) for every word the stage vocalized, in order, edge punctuation trimmed."""
    return [
        (a.strip(_EDGE_PUNCT), b.strip(_EDGE_PUNCT))
        for a, b in _word_pairs(original, vocalized)
        if a != b
    ]


def format_changes(changes: list[Change]) -> str:
    """'original<TAB>vocalized' per line."""
    return "".join(f"{original}\t{vocalized}\n" for original, vocalized in changes)


def parse_vocalized_response(script_text: str, vocalized: str) -> tuple[str, list[Change]]:
    """
    Validates an LLM response against the input script. Returns (text, changes); raises
    PronunciationError if the output is invalid. Shared by sync and batch mode.
    """
    reason = validate_vocalized_output(script_text, vocalized)
    if reason:
        raise PronunciationError(f"pronunciation output rejected: {reason}")
    vocalized = vocalized.replace("\r\n", "\n")
    changes = changed_words(script_text, vocalized)
    logger.info(f"Pronunciation stage: {len(changes)} word(s) vocalized.")
    return vocalized, changes


def vocalize_pronunciation(script_text: str, generate: Callable[[str], str]) -> tuple[str, list[Change]]:
    """
    Sends the script to the LLM via `generate(script_text) -> text` and validates.
    Returns (text, changes). Raises PronunciationError if the call fails or the output is invalid.
    """
    try:
        vocalized = generate(script_text)
    except Exception as e:
        raise PronunciationError(f"pronunciation LLM call failed: {e}") from e
    return parse_vocalized_response(script_text, vocalized)


class PronunciationVocalizer:
    """
    The pronunciation stage, bound to its own generator (model, temperature), instruction
    and thinking_level from config.yaml. The same request and the same validation are used
    for a synchronous call and for a batch job.
    """

    def __init__(self, generator, instruction: str, thinking_level: Optional[str]):
        self.generator = generator
        self.instruction = instruction
        self.thinking_level = thinking_level

    @property
    def supports_batch(self) -> bool:
        from pipeline.generator import BatchCapableGenerator
        return isinstance(self.generator, BatchCapableGenerator)

    def vocalize(self, script_text: str) -> tuple[str, list[Change]]:
        """Synchronous call. Raises PronunciationError on any failure."""
        return self.vocalize_with_usage(script_text)[0]

    def vocalize_with_usage(self, script_text: str):
        """Like vocalize(), also returning the TokenUsage of the call: ((text, changes), usage)."""
        results = []

        def generate(text: str) -> str:
            results.append(self.generator.complete(
                self.instruction, text, self.generator.temperature, self.thinking_level
            ))
            return results[0].text

        return vocalize_pronunciation(script_text, generate), results[0].usage

    def submit_batch(self, script_text: str, custom_id: str, cache_dir: str) -> str:
        """Submits the request as a batch job; returns the job id."""
        return self.generator.submit_batch(
            system_instruction=self.instruction,
            user_prompt=script_text,
            temperature=self.generator.temperature,
            custom_id=custom_id,
            cache_dir=cache_dir,
            thinking_level=self.thinking_level,
        )

    def batch_result(self, script_text: str, job_id: str) -> tuple[str, list[Change]]:
        """Fetches and validates a completed batch job. Raises PronunciationError on failure."""
        try:
            vocalized = self.generator.get_batch_result(job_id)
        except Exception as e:
            raise PronunciationError(f"pronunciation batch result unavailable: {e}") from e
        return parse_vocalized_response(script_text, vocalized)


def create_pronunciation_vocalizer(config, generator_factory=None) -> PronunciationVocalizer:
    """Builds the stage from the 'pronunciation' section of config.yaml (ignores pronunciation.enabled)."""
    if generator_factory is None:
        from pipeline.factory import create_pronunciation_generator as generator_factory
    generator = generator_factory(config)
    logger.info(f"Pronunciation stage via {config.pronunciation_engine} ({generator.model_name}).")
    return PronunciationVocalizer(generator, config.pronunciation_instruction, config.pronunciation_thinking_level)
