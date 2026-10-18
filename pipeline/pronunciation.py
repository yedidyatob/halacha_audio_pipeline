"""
Pronunciation stage: an LLM adds nikkud ONLY to words the TTS engine could read in more
than one way (context-dependent homographs such as עובר / פת / סופר, and letter names used
as letters). The output is plain Hebrew text; there is no markup.

It runs after the dictionary nikkud stage (apply_nikkud), so its input may already contain
vocalized words; the LLM must leave those exactly as they are.

Validation (strict), word by word:
  - words, whitespace and lines must be unchanged (only trailing whitespace and line endings
    may differ);
  - a word that already had nikkud in the input must be byte-identical in the output;
  - any other word, with nikkud removed, must equal the input word, except that ו / י may be
    added or removed (standard voweled spelling, e.g. עובר -> עֻבָּר, הקילו -> הֵקֵלּוּ) anywhere
    but at the start of the word;
  - no markup, no empty output.
A violation or an LLM call failure fails the attempt; it is retried with a new synchronous
call up to pronunciation.max_retries times (pipeline.retry; each failure is logged), then
PronunciationError is raised with the last reason. There is no fallback: the caller decides what
to do (the pipeline fails that siman).
"""

from __future__ import annotations

import difflib
import re
from typing import Callable, Optional

from pipeline.logger import get_logger
from pipeline.retry import call_with_retries

logger = get_logger(__name__)

# Hebrew points and cantillation marks. Excludes maqaf (U+05BE), paseq (U+05C0),
# sof pasuq (U+05C3) and nun hafukha (U+05C6), which are punctuation, not nikkud.
_NIKKUD_RE = re.compile("[\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]")

# Markup the LLM must never emit (e.g. leftover SSML / XML / HTML tags).
_MARKUP_RE = re.compile(r"</?[A-Za-z][^<>]*>")

# A ו / י the LLM may add or remove (matres lectionis, for standard voweled spelling): one
# that follows a Hebrew letter, or a geresh / gershayim that follows one (abbreviations), so
# never the first letter of a word (e.g. the conjunction ו).
_MATRES_RE = re.compile("(?<=[\u05D0-\u05EA])[\u05D5\u05D9]|(?<=[\u05D0-\u05EA][\"'\u05F3\u05F4])[\u05D5\u05D9]")
_ANY_VAV_YOD_RE = re.compile("[\u05D5\u05D9]")

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


def _layout(text: str) -> list[str]:
    """The whitespace between words (and at the ends): equal layouts mean the same words per line."""
    return re.split(r"\S+", text)


def _word_pairs(original: str, vocalized: str) -> list[Change]:
    """
    Aligned (input word, output word) pairs. Only valid once both texts are known to have the
    same layout, so whitespace splitting yields the same number of words.
    """
    return list(zip(_normalize(original).split(), _normalize(vocalized).split()))


def _word_error(original: str, output: str) -> Optional[str]:
    """Why `output` is not a valid rendering of the input word `original`, or None if it is."""
    if has_nikkud(original):
        return None if output == original else "already vocalized, must stay identical"
    letters = strip_nikkud(output)
    if _MATRES_RE.sub("", letters) == _MATRES_RE.sub("", original):
        return None
    if _ANY_VAV_YOD_RE.sub("", letters) == _ANY_VAV_YOD_RE.sub("", original):
        return "ו/י added or removed at the start of the word"
    return "letters changed beyond adding/removing ו/י"


def validate_vocalized_output(original: str, vocalized: str) -> Optional[str]:
    """Returns None if `vocalized` is a valid output for `original`, else a reason string."""
    if not vocalized or not vocalized.strip():
        return "LLM returned empty output"
    markup = [m for m in find_markup(vocalized) if m not in original]
    if markup:
        return f"markup in output: {markup[:5]}"
    expected = strip_nikkud(_normalize(original))
    actual = strip_nikkud(_normalize(vocalized))
    if _layout(expected) != _layout(actual):
        return "text changed (ignoring nikkud): " + _diff_summary(expected, actual)
    rejected = [
        f"{a} -> {b} ({error})"
        for a, b in _word_pairs(original, vocalized)
        if (error := _word_error(a, b))
    ]
    if rejected:
        more = f" ... (+{len(rejected) - 5} more)" if len(rejected) > 5 else ""
        return f"{len(rejected)} word(s) changed invalidly: " + " | ".join(rejected[:5]) + more
    return None


def changed_words(original: str, vocalized: str) -> list[Change]:
    """
    (original, vocalized) for every word the stage changed (vocalized, and possibly respelled
    with ו / י), in order, edge punctuation trimmed.
    """
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
    respelled = sum(strip_nikkud(b) != a for a, b in changes)
    logger.info(f"Pronunciation stage: {len(changes)} word(s) vocalized ({respelled} respelled with ו/י).")
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
    The pronunciation stage, bound to its own generator (model, temperature), instruction,
    thinking_level and max_retries from config.yaml. The same request and the same validation
    are used for a synchronous call and for a batch job. vocalize_once and batch_result are
    single attempts; the pipeline retries them (main.run_llm_stage), vocalize does it itself.
    """

    def __init__(self, generator, instruction: str, thinking_level: Optional[str], max_retries: int):
        self.generator = generator
        self.instruction = instruction
        self.thinking_level = thinking_level
        self.max_retries = max_retries

    @property
    def supports_batch(self) -> bool:
        from pipeline.generator import BatchCapableGenerator
        return isinstance(self.generator, BatchCapableGenerator)

    def vocalize(self, script_text: str) -> tuple[str, list[Change]]:
        """Synchronous call with up to max_retries retries. Raises PronunciationError if every attempt fails."""
        return call_with_retries(lambda: self.vocalize_once(script_text), self.max_retries, "Pronunciation",
                                 retry_on=PronunciationError, error_type=PronunciationError)

    def vocalize_once(self, script_text: str) -> tuple[str, list[Change]]:
        """One synchronous call. Raises PronunciationError if it fails or the output is invalid."""
        return vocalize_pronunciation(
            script_text,
            lambda text: self.generator.generate_text(self.instruction, text, thinking_level=self.thinking_level),
        )

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
        """Fetches and validates a completed batch job (one attempt). Raises PronunciationError on failure."""
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
    return PronunciationVocalizer(generator, config.pronunciation_instruction, config.pronunciation_thinking_level,
                                  config.pronunciation_max_retries)
