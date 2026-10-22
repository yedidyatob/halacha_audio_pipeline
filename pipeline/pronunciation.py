"""
Pronunciation stage: an LLM adds nikkud ONLY to words the TTS engine could read in more
than one way (context-dependent homographs such as עובר / פת / סופר, and letter names used
as letters). The output is plain Hebrew text; there is no markup.

It runs after the dictionary nikkud stage (apply_nikkud), so its input may already contain
vocalized words; the LLM must leave those exactly as they are.

Validation, the same in sync and batch mode.

The whole output is rejected (the attempt fails) when:
  - the call fails, or the output is empty or contains markup;
  - the words do not line up with the input: line count or words per line differ (merged,
    split, missing or extra words; only trailing whitespace and line endings may differ);
  - more than pronunciation.max_reverted_ratio of the words had to be reverted (see below).
A failed attempt is retried up to pronunciation.max_retries times (each failure is logged), then
PronunciationError is raised with the last reason: by the pipeline (main.run_llm_stage: new sync
calls, or new batch jobs in batch mode) or by vocalize (pipeline.retry). There is no fallback:
the caller decides what to do (the pipeline fails that siman).

Otherwise each invalid word is REVERTED: the input word is put back exactly, and the revert is
logged and listed in the review file. A word is invalid when (word_revert_reason):
  - it already had nikkud in the input and is not byte-identical in the output;
  - with nikkud removed, it differs from the input word other than by adding or removing ו / י
    (standard voweled spelling, e.g. עובר -> עֻבָּר, הקילו -> הֵקֵלּוּ), or a ו / י was added or
    removed at its first letter;
  - it is a pronunciation.never_vocalize word and got nikkud.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import NamedTuple, Optional

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

# Words of context on each side of a changed word in the review list of changed words.
CONTEXT_WORDS = 5


class Change(NamedTuple):
    """One word the stage changed (vocalized, and possibly respelled with ו / י)."""
    original: str   # input word, edge punctuation trimmed
    vocalized: str  # output word, edge punctuation trimmed
    line: int       # 1-based line of the word in the vocalized text
    context: str    # up to CONTEXT_WORDS output words on each side (same line), the word in [brackets]


class Revert(NamedTuple):
    """One word whose model version was invalid and was replaced by the input word."""
    original: str   # input word (put back), edge punctuation trimmed
    model: str      # the model's version, edge punctuation trimmed
    reason: str     # word_revert_reason
    line: int       # 1-based line of the word in the output text
    context: str    # up to CONTEXT_WORDS model-output words on each side (same line), the word in [brackets]


class Vocalized(NamedTuple):
    """An accepted pronunciation result: the text (reverts applied) and its review lists."""
    text: str
    changes: list   # [Change], the words vocalized in text
    reverted: list  # [Revert], the model's invalid words that were put back


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


def word_vocalization_error(original: str, output: str) -> Optional[str]:
    """Why `output` is not a valid rendering of the input word `original`, or None if it is."""
    if has_nikkud(original):
        return None if output == original else "already vocalized, must stay identical"
    letters = strip_nikkud(output)
    if _MATRES_RE.sub("", letters) == _MATRES_RE.sub("", original):
        return None
    if _ANY_VAV_YOD_RE.sub("", letters) == _ANY_VAV_YOD_RE.sub("", original):
        return "ו/י added or removed at the start of the word"
    return "letters changed beyond adding/removing ו/י"


NEVER_VOCALIZE_REASON = "never_vocalize word must not get nikkud"


def word_revert_reason(original: str, output: str, never_vocalize: frozenset) -> Optional[str]:
    """Why the model's `output` for the input word `original` must be reverted, or None if it may stay."""
    if output == original:
        return None
    error = word_vocalization_error(original, output)
    if error:
        return error
    if original.strip(_EDGE_PUNCT) in never_vocalize:
        return NEVER_VOCALIZE_REASON
    return None


def validate_layout(original: str, vocalized: str) -> Optional[str]:
    """
    None if the output's words line up with the input's (same lines, words and whitespace between
    them; trailing whitespace and line endings aside), has no markup and is not empty, else the
    reason the whole output is rejected.
    """
    if not vocalized or not vocalized.strip():
        return "LLM returned empty output"
    markup = [m for m in find_markup(vocalized) if m not in original]
    if markup:
        return f"markup in output: {markup[:5]}"
    expected, actual = strip_nikkud(_normalize(original)), strip_nikkud(_normalize(vocalized))
    if _layout(expected) != _layout(actual):
        return "words do not line up with the input (ignoring nikkud): " + _diff_summary(expected, actual)
    return None


def revert_invalid_words(original: str, vocalized: str, never_vocalize: frozenset) -> tuple[str, list[Revert]]:
    """
    `vocalized` with every invalid word (word_revert_reason) replaced by the input word, and the
    reverts. Only valid once validate_layout passed. The output's whitespace is kept as written.
    """
    raw_lines = vocalized.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # _normalize drops leading blank lines; line numbers refer to the output text as written
    first = next((n for n, line in enumerate(raw_lines) if line.strip()), 0)
    reverts = []
    for offset, original_line in enumerate(_normalize(original).split("\n")):
        index = first + offset
        parts = re.split(r"(\s+)", raw_lines[index])  # words at the even indexes ("" at the ends)
        word_slots = [i for i in range(0, len(parts), 2) if parts[i]]
        output_words = [parts[i] for i in word_slots]
        for k, (slot, a) in enumerate(zip(word_slots, original_line.split())):
            reason = word_revert_reason(a, parts[slot], never_vocalize)
            if reason:
                reverts.append(Revert(a.strip(_EDGE_PUNCT), parts[slot].strip(_EDGE_PUNCT), reason, index + 1,
                                      _marked_context(output_words, k)))
                parts[slot] = a
        raw_lines[index] = "".join(parts)
    return "\n".join(raw_lines), reverts


def _marked_context(words: list[str], index: int) -> str:
    """words[index] in [brackets] (edge punctuation outside) with CONTEXT_WORDS words on each side."""
    word = words[index]
    core = word.strip(_EDGE_PUNCT)
    lead = word[:len(word) - len(word.lstrip(_EDGE_PUNCT))]
    trail = word[len(lead) + len(core):]
    start, end = max(0, index - CONTEXT_WORDS), index + 1 + CONTEXT_WORDS
    parts = words[start:index] + [f"{lead}[{core}]{trail}"] + words[index + 1:end]
    return ("..." if start > 0 else "") + " ".join(parts) + ("..." if end < len(words) else "")


def changed_words(original: str, vocalized: str) -> list[Change]:
    """
    Every word the stage changed, in order (see Change). Only valid once the texts are known to
    line up (validate_layout).
    """
    raw_lines = vocalized.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # _normalize drops leading blank lines; line numbers refer to the vocalized text as written
    first_line = next((n for n, line in enumerate(raw_lines, 1) if line.strip()), 1)
    changes = []
    lines = zip(_normalize(original).split("\n"), _normalize(vocalized).split("\n"))
    for line_no, (original_line, vocalized_line) in enumerate(lines, first_line):
        output_words = vocalized_line.split()
        for i, (a, b) in enumerate(zip(original_line.split(), output_words)):
            if a != b:
                changes.append(Change(a.strip(_EDGE_PUNCT), b.strip(_EDGE_PUNCT), line_no,
                                      _marked_context(output_words, i)))
    return changes


CHANGES_HEADER = "varies\toriginal\tvocalized\tcount\tfirst_line\tcontext"


REVERTED_TITLE = "reverted (the model's version was invalid; the original word was kept)"
REVERTED_HEADER = "original\tmodel\treason\tline\tcontext"


def format_review(result: Vocalized) -> str:
    """
    The review file: the changed words (format_changes), a blank line, then the reverted section:
    a REVERTED_TITLE line, a REVERTED_HEADER line and one tab-separated line per reverted word,
    in text order (the context shows the model's version in [brackets]).
    """
    rows = [REVERTED_TITLE, REVERTED_HEADER]
    rows += [f"{r.original}\t{r.model}\t{r.reason}\t{r.line}\t{r.context}" for r in result.reverted]
    return format_changes(result.changes) + "\n" + "\n".join(rows) + "\n"


def format_changes(changes: list[Change]) -> str:
    """
    The review list of changed words: a CHANGES_HEADER line, then one tab-separated line per
    distinct (original, vocalized) pair with its count, first line and the context of its first
    occurrence. varies is "yes" when the same original word got more than one vocalization; those
    words come first (each variant on its own line, kept together), then the rest by count.
    """
    pairs: dict[tuple[str, str], list] = {}  # (original, vocalized) -> [count, first Change]
    for change in changes:
        pairs.setdefault((change.original, change.vocalized), [0, change])[0] += 1
    variants = Counter(original for original, _ in pairs)
    totals = Counter(change.original for change in changes)
    first_lines = {}
    for change in changes:
        first_lines.setdefault(change.original, change.line)

    def order(item):
        (original, _), (count, first) = item
        return (variants[original] == 1, -totals[original], first_lines[original], -count, first.line)

    rows = [CHANGES_HEADER]
    for (original, vocalized), (count, first) in sorted(pairs.items(), key=order):
        varies = "yes" if variants[original] > 1 else "no"
        rows.append(f"{varies}\t{original}\t{vocalized}\t{count}\t{first.line}\t{first.context}")
    return "\n".join(rows) + "\n"


def parse_vocalized_response(script_text: str, vocalized: str, never_vocalize: frozenset,
                             max_reverted_ratio: float) -> Vocalized:
    """
    Validates an LLM response against the input script and reverts its invalid words (see the
    module docstring). Raises PronunciationError if the whole output is rejected. Shared by sync
    and batch mode.
    """
    reason = validate_layout(script_text, vocalized)
    if reason:
        raise PronunciationError(f"pronunciation output rejected: {reason}")
    text, reverted = revert_invalid_words(script_text, vocalized, never_vocalize)
    total = len(_normalize(script_text).split())
    if len(reverted) > max_reverted_ratio * total:
        shown = " | ".join(f"{r.original} -> {r.model} ({r.reason})" for r in reverted[:5])
        more = f" ... (+{len(reverted) - 5} more)" if len(reverted) > 5 else ""
        raise PronunciationError(
            f"pronunciation output rejected: {len(reverted)} of {total} word(s) invalid, more than "
            f"max_reverted_ratio {max_reverted_ratio}: {shown}{more}")
    for r in reverted:
        logger.warning(f"Pronunciation stage: reverted {r.model} -> {r.original} on line {r.line} ({r.reason}).")
    changes = changed_words(script_text, text)
    respelled = sum(strip_nikkud(c.vocalized) != c.original for c in changes)
    logger.info(f"Pronunciation stage: {len(changes)} word(s) vocalized ({respelled} respelled with ו/י), "
                f"{len(reverted)} reverted.")
    return Vocalized(text, changes, reverted)


class PronunciationVocalizer:
    """
    The pronunciation stage, bound to its own generator (model, temperature), instruction,
    thinking_level and max_retries from config.yaml. The same request and the same validation
    are used for a synchronous call and for a batch job. vocalize_once and batch_result are
    single attempts; the pipeline retries them (main.run_llm_stage: new calls in sync mode, new
    batch jobs in batch mode), vocalize retries itself.
    """

    def __init__(self, generator, instruction: str, thinking_level: Optional[str], max_retries: int,
                 never_vocalize: list, max_reverted_ratio: float):
        self.generator = generator
        self.instruction = instruction
        self.thinking_level = thinking_level
        self.max_retries = max_retries
        self.never_vocalize = frozenset(never_vocalize)
        self.max_reverted_ratio = max_reverted_ratio

    def parse(self, script_text: str, vocalized: str) -> Vocalized:
        """Validates a response (sync or batch); raises PronunciationError if it is rejected."""
        return parse_vocalized_response(script_text, vocalized, self.never_vocalize, self.max_reverted_ratio)

    @property
    def supports_batch(self) -> bool:
        from pipeline.generator import BatchCapableGenerator
        return isinstance(self.generator, BatchCapableGenerator)

    def vocalize(self, script_text: str) -> Vocalized:
        """Synchronous call with up to max_retries retries. Raises PronunciationError if every attempt fails."""
        return call_with_retries(lambda: self.vocalize_once(script_text), self.max_retries, "Pronunciation",
                                 retry_on=PronunciationError, error_type=PronunciationError)

    def vocalize_once(self, script_text: str) -> Vocalized:
        """One synchronous call. Raises PronunciationError if it fails or the output is rejected."""
        try:
            vocalized = self.generator.generate_text(self.instruction, script_text, thinking_level=self.thinking_level)
        except Exception as e:
            raise PronunciationError(f"pronunciation LLM call failed: {e}") from e
        return self.parse(script_text, vocalized)

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

    def batch_result(self, script_text: str, job_id: str) -> Vocalized:
        """
        Fetches and validates a completed batch job (one attempt). Raises BatchResultError if the
        job has no usable result or its output is rejected; any other exception means the fetch
        itself failed (the result may still be there).
        """
        from pipeline.generator import BatchResultError
        vocalized = self.generator.get_batch_result(job_id)
        try:
            return self.parse(script_text, vocalized)
        except PronunciationError as e:
            raise BatchResultError(str(e)) from e


def create_pronunciation_vocalizer(config, generator_factory=None) -> PronunciationVocalizer:
    """Builds the stage from the 'pronunciation' section of config.yaml (ignores pronunciation.enabled)."""
    if generator_factory is None:
        from pipeline.factory import create_pronunciation_generator as generator_factory
    generator = generator_factory(config)
    logger.info(f"Pronunciation stage via {config.pronunciation_engine} ({generator.model_name}).")
    return PronunciationVocalizer(generator, config.pronunciation_instruction, config.pronunciation_thinking_level,
                                  config.pronunciation_max_retries, config.pronunciation_never_vocalize,
                                  config.pronunciation_max_reverted_ratio)
