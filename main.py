import os
import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from dotenv import load_dotenv

# Load environment variables from .env file at startup
from pathlib import Path

# Get the path to the directory where the current script resides
current_dir = Path(__file__).resolve().parent

# Build the path to the .env file one level up
dotenv_path = current_dir.parent / ".env"

# Load the file
load_dotenv(dotenv_path=dotenv_path)

from pipeline.config import PipelineConfig
from pipeline.input_parser import parse_simanim_string
from pipeline.extractor import SefariaExtractor
from pipeline.logger import get_logger
from pipeline.utils import save_output_file
from pipeline.factory import create_generator_engine, create_tts_engine
from pipeline.generator import BatchResultError
from pipeline.gematria import int_to_gematria
from pipeline.apply_nikkud import apply_nikkud_with_overrides
from pipeline.nikkud_review import format_review

logger = get_logger("halacha_pipeline_cli")

STAGE_DRAFTS = "Stage 1 (Drafts)"
STAGE_RELATIONS = "Stage 2 (Relations)"
STAGE_POLISH = "Stage 3 (Polishing and nikkud)"
STAGE_NIKKUD = "Stage 3.1 (Dictionary nikkud)"
STAGE_TRANSCRIPT = "Stage 3.2 (Transcript save)"
STAGE_TTS = "TTS"


class SimanStageError(Exception):
    """A single siman failed at a named stage; the run continues with the next siman."""

    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


def item_label(key) -> str:
    """'Siman 94' for a siman, the key itself for a non-siman item (the Stage 2 relations map)."""
    return f"Siman {key}" if isinstance(key, int) else str(key)


def run_siman_stage(siman: int, stage: str, fn):
    """Runs fn(); any exception is logged and re-raised as SimanStageError(stage, reason)."""
    try:
        return fn()
    except SimanStageError:
        raise
    except Exception as e:
        logger.error(f"{item_label(siman)} failed at {stage}: {e}")
        raise SimanStageError(stage, str(e)) from e


class RunSummary:
    """Collects per-siman outcomes for the end-of-run summary."""

    def __init__(self):
        self.succeeded: list[int] = []
        self.failed: dict[int, tuple[str, str]] = {}

    def success(self, siman: int) -> None:
        self.succeeded.append(siman)

    def failure(self, siman: int, stage: str, reason: str) -> None:
        self.failed[siman] = (stage, reason)

    @property
    def has_failures(self) -> bool:
        return bool(self.failed)

    def format(self) -> str:
        total = len(self.succeeded) + len(self.failed)
        lines = [
            "===== Run summary =====",
            f"Succeeded ({len(self.succeeded)}/{total}): "
            + (", ".join(str(s) for s in self.succeeded) or "none"),
            f"Failed ({len(self.failed)}/{total}): "
            + (", ".join(str(s) for s in self.failed) or "none"),
        ]
        for siman, (stage, reason) in self.failed.items():
            lines.append(f"  - {item_label(siman)} | stage: {stage} | reason: {reason}")
        return "\n".join(lines)

    def log(self) -> None:
        """Writes the summary through the pipeline logger (stdout)."""
        log_fn = logger.error if self.has_failures else logger.info
        for line in self.format().split("\n"):
            log_fn(line)


def run_per_siman(summary: RunSummary, items: dict, stage: str, fn) -> dict:
    """
    Applies fn(siman, value) to every {siman: value}. Returns {siman: result} for the
    simanim that succeeded; each failure is recorded in the summary under its stage.
    """
    results = {}
    for siman, value in items.items():
        try:
            results[siman] = run_siman_stage(siman, stage, lambda: fn(siman, value))
        except SimanStageError as e:
            summary.failure(siman, e.stage, e.reason)
    return results


@dataclass(frozen=True)
class BatchCalls:
    """How an LLM stage runs in batch mode: submit(key, value) -> job_id, fetch(key, value, job_id) -> result."""
    generator: Any
    submit: Callable[[Any, Any], str]
    fetch: Callable[[Any, Any, str], Any]


def _sync_round(call, items: dict, accept) -> dict:
    """One synchronous call(key, value) per item; accept(key, result) on success. Returns {key: error}."""
    errors = {}
    for key, value in items.items():
        try:
            result = call(key, value)
        except Exception as e:
            errors[key] = e
        else:
            accept(key, result)
    return errors


class BatchTimeoutError(TimeoutError):
    """--batch-timeout expired with jobs still unresolved; the message lists their job ids."""


def _batch_round(batch: BatchCalls, items: dict, stage: str, args, accept) -> dict:
    """
    One batch job per item, all polled together; accept(key, result) as each result is fetched.
    Returns {key: error} for failed submits, jobs the API reports as failed, and finished jobs
    without a usable result (BatchResultError) - the attempts a new batch job may fix.
    """
    errors, jobs = {}, {}
    for key, value in items.items():
        try:
            jobs[key] = batch.submit(key, value)
        except Exception as e:
            errors[key] = RuntimeError(f"batch submit failed: {e}")
        else:
            logger.info(f"Submitted {stage} batch job for {item_label(key)}. Job ID: {jobs[key]}")
    if jobs:
        errors.update(await_batch_jobs(batch, jobs, items, stage, args, accept))
    return errors


def await_batch_jobs(batch: BatchCalls, jobs: dict, items: dict, stage: str, args, accept) -> dict:
    """
    Polls {key: job_id} every --batch-poll-interval seconds; a completed job's result is fetched
    right away and passed to accept(key, result). Returns {key: error} for jobs the API reports
    as failed and completed jobs whose result is unusable (BatchResultError).
    A failed status check or result fetch (network / API error) says nothing about the job: it
    is logged and retried at the next poll, never resubmitted. Raises BatchTimeoutError listing
    every unresolved job id when --batch-timeout expires.
    """
    poll_interval, timeout = args.batch_poll_interval, args.batch_timeout
    start_time = time.time()
    pending = dict(jobs)
    unknown = {}  # key -> last status-check / fetch error, while it is still unresolved
    errors = {}
    logger.info(f"Polling {len(pending)} {stage} batch job(s) (interval: {poll_interval}s, timeout: {timeout}s)...")

    while True:
        for key, job_id in list(pending.items()):
            label = f"{stage} batch job for {item_label(key)} (job_id: {job_id})"
            try:
                status = batch.generator.get_batch_status(job_id)
            except Exception as e:
                unknown[key] = f"status check failed: {e}"
                logger.warning(f"{label}: status check failed: {e}; status unknown, checking again at the next poll.")
                continue
            if status == "pending":
                unknown.pop(key, None)
                continue
            if status == "failed":
                logger.error(f"\u274c {label} failed.")
                errors[key] = RuntimeError(f"batch job failed (job_id: {job_id})")
            else:
                try:
                    result = batch.fetch(key, items[key], job_id)
                except BatchResultError as e:
                    logger.error(f"\u274c {label} completed without a usable result: {e}")
                    errors[key] = e
                except Exception as e:
                    unknown[key] = f"completed, result fetch failed: {e}"
                    logger.warning(f"{label}: completed, but fetching its result failed: {e}; "
                                   f"fetching again at the next poll.")
                    continue
                else:
                    logger.info(f"\u2705 {label} completed.")
                    accept(key, result)
            unknown.pop(key, None)
            del pending[key]

        if not pending:
            break
        elapsed = time.time() - start_time
        if elapsed > timeout:
            unresolved = "; ".join(f"{item_label(key)}: job_id {job_id}"
                                   + (f" ({unknown[key]})" if key in unknown else " (still pending)")
                                   for key, job_id in pending.items())
            raise BatchTimeoutError(f"{stage}: --batch-timeout ({timeout}s) expired with {len(pending)} "
                                    f"unresolved batch job(s); fetch them later by job id: {unresolved}")
        logger.info(f"\u23f3 {stage}: {len(pending)} job(s) still pending, {len(jobs) - len(pending)} done. "
                    f"Elapsed: {int(elapsed / 60)}m. Polling again in {poll_interval}s...")
        time.sleep(poll_interval)

    if errors:
        logger.error(f"{stage}: {len(errors)} of {len(jobs)} batch job(s) failed: {list(errors)}")
    else:
        logger.info(f"{stage}: All {len(jobs)} batch job(s) completed successfully.")
    return errors


def run_llm_stage(summary: RunSummary, items: dict, stage: str, call, max_retries: int, args,
                  batch: BatchCalls, on_result: Optional[Callable[[Any, Any], None]] = None) -> dict:
    """
    Runs an LLM stage over {key: value} in rounds: the first round covers every item, and each of
    up to max_retries further rounds covers the items that failed so far. Sync mode (call(key, value))
    retries with new sync calls; batch mode (args.batch) resubmits them all as new batch jobs and
    polls again. Each failed attempt is logged as a warning with its reason. on_result(key, result)
    runs as soon as an item succeeds (e.g. to save it); its own failure fails the item without a retry.
    Returns {key: result} for the items that succeeded; the others are recorded in the summary.
    """
    results = {}

    def accept(key, result):
        try:
            if on_result is not None:
                run_siman_stage(key, stage, lambda: on_result(key, result))
            results[key] = result
        except SimanStageError as e:
            summary.failure(key, e.stage, e.reason)

    attempts = max_retries + 1
    pending = dict(items)
    for number in range(1, attempts + 1):
        if number > 1:
            logger.info(f"{stage}: retry round {number - 1}/{max_retries} for {len(pending)} item(s)"
                        + (" as new batch jobs." if args.batch else "."))
        if args.batch:
            errors = _batch_round(batch, pending, stage, args, accept)
        else:
            errors = _sync_round(call, pending, accept)
        errors = {key: errors[key] for key in items if key in errors}  # item order
        for key, error in errors.items():
            logger.warning(f"{item_label(key)} {stage} attempt {number}/{attempts} failed: {error}"
                           + ("; retrying." if number < attempts else ""))
        pending = {key: items[key] for key in errors}
        if not pending:
            break
    for key, error in errors.items():
        reason = f"{error} (after {attempts} attempt(s))"
        logger.error(f"{item_label(key)} failed at {stage}: {reason}")
        summary.failure(key, stage, reason)
    return {key: results[key] for key in items if key in results}


def abort_if_failed(summary: RunSummary, stage: str, note: str = "") -> None:
    """Stages 1 and 2 are needed by every siman: any failure ends the run (after saving the rest)."""
    if not summary.has_failures:
        return
    summary.log()
    logger.error(f"{stage} failed; aborting the run." + (f" {note}" if note else ""))
    sys.exit(1)


AUDIO_EXTENSION = "mp3"


def model_suffix_for(generator) -> str:
    """The generator's model name as it appears in output and cache file names."""
    return generator.model_name.replace('.', '_')


def lesson_base_name(config, siman: int, model_suffix: str) -> str:
    """Base name of a siman's outputs in output_dir (transcript, nikkud review, audio)."""
    return f"{config.section_slug}_Siman_{siman}_{model_suffix}"


def polish_base_name(config, siman: int, model_suffix: str) -> str:
    """Stage 3 output in output_dir; also its cache (a later run loads it instead of polishing again)."""
    return f"{lesson_base_name(config, siman, model_suffix)}_polish"


def draft_base_name(config, siman: int, model_suffix: str) -> str:
    return f"{config.section_slug}_Siman_{siman}_{model_suffix}_draft"


def relations_base_name(config, context_simanim: list, model_suffix: str) -> str:
    range_str = "_".join(str(s) for s in context_simanim)
    return f"{config.section_slug}_Relations_{range_str}_{model_suffix}"


def save_draft(siman: int, draft_text: str, config, model_suffix: str) -> None:
    save_output_file(
        directory=config.drafts_dir,
        base_name=draft_base_name(config, siman, model_suffix),
        extension="txt",
        content=draft_text,
        logger=logger,
        label=f"{STAGE_DRAFTS} output for Siman {siman}"
    )


def save_polish(siman: int, polished_text: str, config, model_suffix: str) -> None:
    """Stage 3 output (before the nikkud stages); polish_simanim saves each text as soon as it is ready."""
    save_output_file(
        directory=config.output_dir,
        base_name=polish_base_name(config, siman, model_suffix),
        extension="txt",
        content=polished_text,
        logger=logger,
        label=f"{STAGE_POLISH} output for Siman {siman}"
    )


def generate_drafts(simanim: list, extractor, generator, config, args, model_suffix: str) -> dict:
    """
    Stage 1: {siman: draft} for the given simanim (sync or batch, with generator.max_retries).
    Each draft is saved to the cache as soon as it is ready; if any siman still fails, the run
    is aborted after all of them were tried.
    """
    summary = RunSummary()
    contexts = run_per_siman(
        summary, {siman: siman for siman in simanim}, STAGE_DRAFTS,
        lambda siman, _: extractor.compile_simanim_context([siman], target_simanim=[siman])
    )
    drafts = run_llm_stage(
        summary, contexts, STAGE_DRAFTS,
        call=lambda siman, context: generator.generate_siman_script(
            siman=siman, master_context=context,
            system_instruction=config.gemini_system_instruction,
            thinking_level=config.draft_thinking_level,
            thinking_name=draft_base_name(config, siman, model_suffix),
        ),
        max_retries=config.generator_max_retries, args=args,
        batch=BatchCalls(
            generator,
            submit=lambda siman, context: generator.submit_batch(
                system_instruction=config.gemini_system_instruction,
                user_prompt=generator._get_generation_user_prompt(int_to_gematria(siman), context),
                temperature=generator.temperature,
                custom_id=f"stage1_siman_{siman}",
                cache_dir=config.cache_dir,
                thinking_level=config.draft_thinking_level,
            ),
            fetch=lambda siman, context, job_id: generator.get_batch_result(job_id),
        ),
        on_result=lambda siman, draft: save_draft(siman, draft, config, model_suffix),
    )
    abort_if_failed(summary, STAGE_DRAFTS,
                    f"{len(drafts)} draft(s) were saved to the cache; a rerun only regenerates the missing ones.")
    return drafts


RELATIONS_KEY = "Relations map"


def generate_relations(drafts: dict, generator, config, args, base_name: str) -> str:
    """
    Stage 2: the cross-siman relations map (sync or batch, with generator.max_retries). Aborts on failure.
    base_name is the map's output base name (names the sync call's thinking file).
    """
    summary = RunSummary()
    result = run_llm_stage(
        summary, {RELATIONS_KEY: drafts}, STAGE_RELATIONS,
        call=lambda _, drafts: generator.analyze_cross_relations(drafts, config.relations_instruction,
                                                               thinking_name=base_name),
        max_retries=config.generator_max_retries, args=args,
        batch=BatchCalls(
            generator,
            submit=lambda _, drafts: generator.submit_batch(
                system_instruction=config.relations_instruction,
                user_prompt=generator._get_relations_user_prompt(drafts),
                temperature=generator.temperature,
                custom_id="stage2_relations",
                cache_dir=config.cache_dir
            ),
            fetch=lambda _, drafts, job_id: generator.get_batch_result(job_id),
        ),
    )
    abort_if_failed(summary, STAGE_RELATIONS)
    return result[RELATIONS_KEY]


def polish_simanim(drafts: dict, relations_text: str, generator, config, args, summary: RunSummary,
                   model_suffix: str) -> dict:
    """Stage 3: {siman: draft} -> {siman: polished_text} (sync or batch, with generator.max_retries)."""
    return run_llm_stage(
        summary, drafts, STAGE_POLISH,
        call=lambda siman, draft: generator.polish_siman_script(
            draft, relations_text, config.polishing_instruction,
            temperature=config.polish_temperature, thinking_name=polish_base_name(config, siman, model_suffix)),
        max_retries=config.generator_max_retries, args=args,
        batch=BatchCalls(
            generator,
            submit=lambda siman, draft: generator.submit_batch(
                system_instruction=config.polishing_instruction,
                user_prompt=generator._get_polishing_user_prompt(draft, relations_text),
                temperature=config.polish_temperature,
                custom_id=f"stage3_siman_{siman}",
                cache_dir=config.cache_dir
            ),
            fetch=lambda siman, draft, job_id: generator.get_batch_result(job_id),
        ),
        on_result=lambda siman, polished_text: save_polish(siman, polished_text, config, model_suffix),
    )


SINGLE_PASS_SUFFIX = "single_pass"


def single_pass_suffix(model_suffix: str) -> str:
    """--single-pass outputs (polish cache, transcript, review, audio) are named apart from the normal run's."""
    return f"{model_suffix}_{SINGLE_PASS_SUFFIX}"


def polish_sources(sources: dict, generator, config, args, summary: RunSummary, model_suffix: str) -> dict:
    """
    --single-pass Stage 3: {siman: raw Sefaria sources} -> {siman: polished_text} in one polish call per siman
    (config single_pass_user_prompt; sync or batch, with generator.max_retries).
    """
    def user_prompt(siman, text):
        return config.format_single_pass_user_prompt(int_to_gematria(siman), text)

    return run_llm_stage(
        summary, sources, STAGE_POLISH,
        call=lambda siman, text: generator.polish_with_prompt(
            user_prompt(siman, text), config.polishing_instruction,
            temperature=config.polish_temperature, thinking_name=polish_base_name(config, siman, model_suffix)),
        max_retries=config.generator_max_retries, args=args,
        batch=BatchCalls(
            generator,
            submit=lambda siman, text: generator.submit_batch(
                system_instruction=config.polishing_instruction,
                user_prompt=user_prompt(siman, text),
                temperature=config.polish_temperature,
                custom_id=f"stage3_single_pass_siman_{siman}",
                cache_dir=config.cache_dir
            ),
            fetch=lambda siman, text, job_id: generator.get_batch_result(job_id),
        ),
        on_result=lambda siman, polished_text: save_polish(siman, polished_text, config, model_suffix),
    )


def apply_dictionary_nikkud(polished: dict, summary: RunSummary) -> dict:
    """
    Stage 3.1: {siman: polished_text} -> {siman: (text, overrides)}: the dictionary nikkud applied
    over the polish's nikkud, and the places where it replaced the model's (apply_nikkud.NikkudOverride).
    """
    return run_per_siman(summary, polished, STAGE_NIKKUD, lambda siman, text: apply_nikkud_with_overrides(text))


class PlanError(ValueError):
    """The run cannot start (e.g. --relations-file not found); the message says why."""


@dataclass
class RunPlan:
    """What a run will load from cache and what it will generate; resolved before any API call."""
    context_simanim: list
    target_simanim: list          # Stage 3+ (after --limit-lessons)
    relations_source: str         # "file" (--relations-file), "cache", "generate" or "none" (--single-pass)
    relations_path: str           # file it is loaded from, or where the generated map is saved
    relations_text: str           # "" when it will be generated
    draft_paths: dict             # {siman: cache path} for every draft needed
    drafts: dict                  # {siman: text} loaded from cache or --draft-file
    draft_files: list             # simanim whose draft comes from --draft-file
    missing_drafts: list          # simanim whose Stage 1 draft will be generated
    polish_paths: dict            # {siman: Stage 3 polish path} for every target siman
    polished: dict                # {siman: text} Stage 3 polish loaded from cache

    @property
    def drafts_needed(self) -> list:
        return list(self.draft_paths)

    @property
    def missing_polish(self) -> list:
        """Target simanim whose Stage 3 polish will be generated."""
        return [siman for siman in self.target_simanim if siman not in self.polished]


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def resolve_run_plan(args, config, simanim_list: list, context_list: list, model_suffix: str) -> RunPlan:
    """
    Resolves the relations map (--relations-file, cache unless --overwrite-cache, or generate),
    the Stage 1 drafts it needs and the target simanim's Stage 3 polish (cached vs to generate),
    reading only local files.
    Raises PlanError if the run cannot start.
    """
    relations_text, relations_source = "", "generate"
    if args.single_pass:
        relations_path, relations_source = "", "none"
    elif args.relations_file:
        relations_path = args.relations_file
        if not os.path.exists(relations_path):
            raise PlanError(f"Relations file not found at: {relations_path}")
        try:
            relations_text = _read_text(relations_path)
        except Exception as e:
            raise PlanError(f"Failed to read relations file: {e}") from e
        relations_source = "file"
        logger.info(f"Loaded pre-existing relations map: {relations_path} ({len(relations_text)} chars).")
    else:
        relations_path = os.path.join(config.relations_dir,
                                      relations_base_name(config, context_list, model_suffix) + ".txt")
        if os.path.exists(relations_path) and not args.overwrite_cache:
            try:
                relations_text = _read_text(relations_path)
                relations_source = "cache"
                logger.info(f"Loaded cached Stage 2 relations map: {relations_path} ({len(relations_text)} chars).")
            except Exception as e:
                logger.warning(f"Failed to read cached relations: {e}. Will regenerate.")

    target_simanim = simanim_list
    if args.limit_lessons is not None:
        if args.limit_lessons <= 0:
            raise PlanError("Limit lessons count must be greater than zero.")
        target_simanim = simanim_list[:args.limit_lessons]
        logger.info(f"[DEBUG LIMIT] Restricting Stage 3 styling to: {target_simanim}")

    if args.single_pass:
        # the raw sources go straight to Stage 3: no drafts
        drafts_needed = []
    elif relations_text:
        # relations are available, so only the target simanim being polished need drafts
        drafts_needed = target_simanim
        logger.info(f"Relations map is already loaded. Resolving drafts only for target Simanim: {drafts_needed}")
    else:
        # generating relations needs a draft for every siman in the context list
        drafts_needed = context_list
        logger.info(f"Relations map needs to be generated. Resolving drafts for entire context range: {drafts_needed}")

    unused = sorted(set(args.draft_file) - set(drafts_needed))
    if unused:
        raise PlanError(f"--draft-file given for Simanim {unused}, which this run does not need drafts for "
                        f"(drafts needed: {drafts_needed}).")

    draft_paths, drafts, missing, draft_files = {}, {}, [], []
    for siman in drafts_needed:
        if siman in args.draft_file:
            draft_path = args.draft_file[siman]
            if not os.path.exists(draft_path):
                raise PlanError(f"Draft file for Siman {siman} not found at: {draft_path}")
            try:
                drafts[siman] = _read_text(draft_path)
            except Exception as e:
                raise PlanError(f"Failed to read draft file for Siman {siman}: {e}") from e
            draft_paths[siman] = draft_path
            draft_files.append(siman)
            logger.info(f"Loaded Stage 1 draft for Siman {siman} from --draft-file {draft_path} "
                        f"({len(drafts[siman])} chars).")
            continue
        draft_path = os.path.join(config.drafts_dir, draft_base_name(config, siman, model_suffix) + ".txt")
        draft_paths[siman] = draft_path
        if os.path.exists(draft_path) and not args.overwrite_cache:
            try:
                drafts[siman] = _read_text(draft_path)
                logger.info(f"Loaded cached Stage 1 draft for Siman {siman} ({len(drafts[siman])} chars).")
                continue
            except Exception as e:
                logger.warning(f"Failed to read cached draft at '{draft_path}': {e}. Will regenerate.")
        missing.append(siman)

    polish_paths, polished = {}, {}
    for siman in target_simanim:
        polish_path = os.path.join(config.output_dir, polish_base_name(config, siman, model_suffix) + ".txt")
        polish_paths[siman] = polish_path
        if os.path.exists(polish_path) and not args.overwrite_cache:
            try:
                text = _read_text(polish_path)
            except Exception as e:
                logger.warning(f"Failed to read cached Stage 3 polish at '{polish_path}': {e}. Will regenerate.")
                continue
            if not text.strip():
                logger.warning(f"Cached Stage 3 polish for Siman {siman} is empty: {polish_path}. Will regenerate.")
                continue
            polished[siman] = text
            logger.info(f"Loaded cached Stage 3 polish for Siman {siman}: {polish_path} ({len(text)} chars).")

    return RunPlan(context_simanim=context_list, target_simanim=target_simanim,
                   relations_source=relations_source, relations_path=relations_path,
                   relations_text=relations_text, draft_paths=draft_paths, drafts=drafts,
                   draft_files=draft_files, missing_drafts=missing, polish_paths=polish_paths, polished=polished)


def format_run_plan(plan: RunPlan, args, config, generator, model_suffix: str) -> str:
    """The --dry-run report: what a real run with these arguments would load, generate and write."""
    calls = "batch jobs" if args.batch else "synchronous calls"
    lines = [
        "===== Dry run: plan (no API / LLM / TTS / Sefaria calls were made) =====",
        f"Mode: {'batch' if args.batch else 'sync'} | generator: {config.generator_engine} {generator.model_name}"
        + (" | --overwrite-cache" if args.overwrite_cache else ""),
        f"Failed LLM calls are retried as new {calls}: Stages 1-3 up to {config.generator_max_retries} time(s)",
        f"Target simanim (Stage 3+): {plan.target_simanim}"
        + (f" (--limit-lessons {args.limit_lessons})" if args.limit_lessons is not None else ""),
        f"Context simanim: {plan.context_simanim}",
    ]
    if args.single_pass:
        lines.append("--single-pass: no Stage 1 drafts and no Stage 2 relations map; each target siman's raw "
                     "Sefaria sources go straight to the Stage 3 polish (one Sefaria fetch per polish to generate)")
    if plan.relations_source == "none":
        pass
    elif plan.relations_source == "file":
        lines.append(f"Stage 2 relations map: loaded from --relations-file {plan.relations_path}")
    elif plan.relations_source == "cache":
        lines.append(f"Stage 2 relations map: loaded from cache {plan.relations_path}")
    elif args.stage_1_only:
        lines.append("Stage 2 relations map: not cached; not generated (--stage-1-only)")
    else:
        lines.append(f"Stage 2 relations map: will be GENERATED ({'a batch job' if args.batch else 'a synchronous call'}) "
                     f"from the drafts of {plan.drafts_needed} and saved to {plan.relations_path}")
    cached = [s for s in plan.drafts_needed if s in plan.drafts and s not in plan.draft_files]
    if not args.single_pass:
        lines += format_draft_plan(plan, cached, calls)
    if args.stage_1_only:
        lines.append("--stage-1-only: the run stops after Stage 1.")
        return "\n".join(lines)
    lines += format_lesson_plan(plan, args, config, calls, model_suffix)
    return "\n".join(lines)


def format_draft_plan(plan: RunPlan, cached: list, calls: str) -> list:
    lines = [f"Stage 1 drafts needed: {plan.drafts_needed}"]
    if plan.draft_files:
        lines.append(f"  from --draft-file ({len(plan.draft_files)}): {plan.draft_files}")
        lines += [f"    Siman {s}: {plan.draft_paths[s]}" for s in plan.draft_files]
    lines.append(f"  from cache ({len(cached)}): {cached or 'none'}")
    lines += [f"    Siman {s}: {plan.draft_paths[s]}" for s in cached]
    lines.append(f"  to be GENERATED ({len(plan.missing_drafts)}, as {calls}, after a Sefaria fetch each): "
                 f"{plan.missing_drafts or 'none'}")
    lines += [f"    Siman {s} -> {plan.draft_paths[s]}" for s in plan.missing_drafts]
    return lines


def format_lesson_plan(plan: RunPlan, args, config, calls: str, model_suffix: str) -> list:
    lines = []
    cached_polish = [s for s in plan.target_simanim if s in plan.polished]
    lines.append(f"Stage 3 polish and nikkud -> 3.1 dictionary nikkud: {plan.target_simanim}")
    lines.append(f"  Stage 3 polish from cache ({len(cached_polish)}): {cached_polish or 'none'}")
    lines += [f"    Siman {s}: {plan.polish_paths[s]}" for s in cached_polish]
    lines.append(f"  Stage 3 polish to be GENERATED ({len(plan.missing_polish)}, as {calls}): "
                 f"{plan.missing_polish or 'none'}")
    lines += [f"    Siman {s} -> {plan.polish_paths[s]}" for s in plan.missing_polish]
    lines += [
        f"Transcripts / TTS (tts.engine {config.tts_engine}{', skipped: --skip-tts' if args.skip_tts else ''}):",
    ]
    for siman in plan.target_simanim:
        base = os.path.join(config.output_dir, lesson_base_name(config, siman, model_suffix))
        lines.append(f"    Siman {siman}: {base}_transcript.txt"
                     + ("" if args.skip_tts else f", {base}.{AUDIO_EXTENSION}"))
    lines.append("(Each saved file also gets a timestamped history copy.)")
    return lines


def save_transcript(siman: int, text: str, overrides: list, config, model_suffix: str) -> None:
    """
    Stage 3.2: saves the final transcript and its nikkud review file (nikkud_review.format_review):
    the vocalized words and the dictionary's overrides of the model's nikkud.
    """
    base_name = lesson_base_name(config, siman, model_suffix)
    save_output_file(
        directory=config.output_dir,
        base_name=f"{base_name}_pronunciation_changes",
        extension="txt",
        content=format_review(text, overrides),
        logger=logger,
        label=f"Nikkud review for Siman {siman}"
    )
    save_output_file(
        directory=config.output_dir,
        base_name=f"{base_name}_transcript",
        extension="txt",
        content=text,
        logger=logger,
        label=f"{STAGE_TRANSCRIPT} output for Siman {siman}"
    )


def synthesize_siman_audio(siman: int, text: str, config, model_suffix: str) -> None:
    """TTS for one siman."""
    tts_engine = create_tts_engine(config)

    # Synthesize directly to the audio files using a write_callback
    def tts_callback(path: str) -> None:
        logger.info(f"Synthesizing script for Siman {siman}...")
        tts_engine.synthesize(text=text, output_path=path)

    save_output_file(
        directory=config.output_dir,
        base_name=lesson_base_name(config, siman, model_suffix),
        extension=AUDIO_EXTENSION,
        content=None,
        logger=logger,
        label=f"{STAGE_TTS} output for Siman {siman}",
        write_callback=tts_callback
    )


def finish_siman(siman: int, text: str, overrides: list, config, args, model_suffix: str) -> None:
    """Stage 3.2 (transcript save) then TTS unless --skip-tts. Raises SimanStageError on failure."""
    run_siman_stage(siman, STAGE_TRANSCRIPT, lambda: save_transcript(siman, text, overrides, config, model_suffix))
    if args.skip_tts:
        logger.info("Skipping TTS synthesis as requested by --skip-tts.")
        return
    run_siman_stage(siman, STAGE_TTS, lambda: synthesize_siman_audio(siman, text, config, model_suffix))


def run_lesson_stages(target_simanim: list, drafts: dict, cached_polish: dict, relations_text: str, generator,
                      config, args, model_suffix: str) -> RunSummary:
    """Stage 3 onwards from the Stage 1 drafts and the relations map (see run_polished_stages)."""
    summary = RunSummary()
    available = {}
    for siman in target_simanim:
        if siman in cached_polish:
            continue
        if siman in drafts:
            available[siman] = drafts[siman]
        else:
            logger.error(f"Cannot polish Siman {siman}: Stage 1 draft is missing.")
            summary.failure(siman, STAGE_POLISH, "Stage 1 draft is missing")
    return run_polished_stages(
        target_simanim, cached_polish, summary, config, args, model_suffix,
        polish=lambda: polish_simanim(available, relations_text, generator, config, args, summary, model_suffix)
        if available else {})


def run_single_pass_stages(target_simanim: list, cached_polish: dict, extractor, generator, config, args,
                           model_suffix: str) -> RunSummary:
    """--single-pass Stage 3 onwards: the raw Sefaria sources of each target siman are polished directly."""
    summary = RunSummary()

    def polish():
        to_polish = [siman for siman in target_simanim if siman not in cached_polish]
        sources = run_per_siman(
            summary, {siman: siman for siman in to_polish}, STAGE_POLISH,
            lambda siman, _: extractor.compile_simanim_context([siman], target_simanim=[siman])
        )
        return polish_sources(sources, generator, config, args, summary, model_suffix) if sources else {}

    return run_polished_stages(target_simanim, cached_polish, summary, config, args, model_suffix, polish)


def run_polished_stages(target_simanim: list, cached_polish: dict, summary: RunSummary, config, args,
                        model_suffix: str, polish) -> RunSummary:
    """
    Stage 3 onwards for every target siman, one stage at a time (sync or batch):
    polish and nikkud (one LLM call) -> dictionary nikkud -> transcript save -> TTS.
    A siman in cached_polish ({siman: text}, see resolve_run_plan) skips the polish call.
    A failure removes only that siman from the later stages.
    polish() makes the Stage 3 calls for the simanim not in cached_polish and returns their {siman: text}.
    """
    if all(siman in cached_polish for siman in target_simanim):
        logger.info(f"{STAGE_POLISH}: no calls needed; every target siman's polish was loaded from cache.")
        new_polish = {}
    else:
        new_polish = polish()
    ready = {**cached_polish, **new_polish}
    polished = {siman: ready[siman] for siman in target_simanim if siman in ready}
    final = apply_dictionary_nikkud(polished, summary)
    finished = run_per_siman(
        summary, final, STAGE_TRANSCRIPT,
        lambda siman, item: finish_siman(siman, *item, config, args, model_suffix)
    )
    for siman in finished:
        summary.success(siman)
    return summary


def parse_draft_file_arg(value: str) -> tuple:
    """'SIMAN=PATH' -> (siman, path); SIMAN is a number or Hebrew gematria, as in --simanim."""
    siman_text, sep, path = value.partition("=")
    if not sep or not siman_text.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"expected SIMAN=PATH, got '{value}'")
    try:
        simanim = parse_simanim_string(siman_text.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    if len(simanim) != 1:
        raise argparse.ArgumentTypeError(f"expected a single siman, got '{siman_text}'")
    return simanim[0], path.strip()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Halacha Audio Lesson Generation Pipeline - End to End."
    )
    parser.add_argument(
        "simanim",
        type=str,
        help="Comma-separated list of Simanim and ranges (e.g. '94' or '94,95-97,100')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--limit-lessons",
        type=int,
        default=None,
        help="Debug Limit: Only generate transcripts/audio for the first N Simanim (while keeping full context)."
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip the Text-to-Speech synthesis stage (only generate text scripts)."
    )
    parser.add_argument(
        "--context-range",
        type=str,
        default=None,
        help="Optional broad range of Simanim to use as context (e.g. '87-111'). If not specified, defaults to the target simanim."
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Batch API to run the full pipeline asynchronously (supports both Gemini and OpenAI). Submits all 3 stages as batch jobs with automatic polling between them."
    )
    parser.add_argument(
        "--batch-poll-interval",
        type=int,
        default=60,
        help="Seconds between batch job status polls (default: 60)."
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=86400,
        help="Maximum seconds to wait for batch jobs before timing out (default: 86400 = 24 hours)."
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Force regeneration of Stage 1 drafts, Stage 2 relations maps and Stage 3 polish outputs, "
             "overwriting any cached files."
    )
    parser.add_argument(
        "--relations-file",
        type=str,
        default=None,
        help="Path to a pre-existing cross-relations map file. If specified, Stage 2 relations analysis is skipped."
    )
    parser.add_argument(
        "--draft-file",
        type=parse_draft_file_arg,
        action="append",
        default=[],
        metavar="SIMAN=PATH",
        help="Use an existing Stage 1 draft file for a siman instead of the cache or a new Stage 1 call "
             "(e.g. a draft made with another model). Repeat for several simanim. Never overwritten."
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        help="Skip Stage 1 drafts and Stage 2 relations: polish each siman's raw Sefaria sources directly "
             "(config single_pass_user_prompt), then dictionary nikkud, transcript and TTS as usual. Outputs are "
             "named ..._<model>_single_pass_... so they never collide with a normal run's."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan (relations map, drafts and Stage 3 polish loaded from cache vs generated, simanim "
             "going through Stage 3/3.1/TTS, file paths) and exit without any API, LLM, TTS or Sefaria call."
    )
    parser.add_argument(
        "--stage-1-only",
        action="store_true",
        help="Only run Stage 1 (Data Extraction) and exit."
    )
    args = parser.parse_args()
    draft_files = {}
    for siman, path in args.draft_file:
        if siman in draft_files:
            parser.error(f"--draft-file given twice for Siman {siman}")
        draft_files[siman] = path
    args.draft_file = draft_files
    if args.single_pass:
        conflicts = [flag for flag, used in (("--stage-1-only", args.stage_1_only),
                                             ("--relations-file", args.relations_file),
                                             ("--draft-file", draft_files),
                                             ("--context-range", args.context_range)) if used]
        if conflicts:
            parser.error(f"--single-pass has no Stage 1 drafts or Stage 2 relations map, so it cannot be combined "
                         f"with {', '.join(conflicts)}")
    return args

def main():
    args = parse_arguments()
    
    # 1. Parse Simanim lists
    try:
        simanim_list = parse_simanim_string(args.simanim)
        logger.info(f"Target Simanim parsed: {simanim_list}")
        
        # Determine context list (all Simanim we need drafts for)
        if args.context_range:
            context_list = parse_simanim_string(args.context_range)
            # Ensure target Simanim are included in the context list
            context_list = sorted(list(set(context_list + simanim_list)))
        else:
            context_list = simanim_list
            
        logger.info(f"Context Simanim parsed: {context_list}")
    except ValueError as e:
        logger.error(f"Input Parsing Error: {e}")
        sys.exit(1)
        
    # 2. Load Configuration
    try:
        config = PipelineConfig(args.config)
    except Exception as e:
        logger.error(f"Configuration Loading Failed: {e}")
        sys.exit(1)
        
    # Determine model suffix for naming
    try:
        generator = create_generator_engine(config)
        model_suffix = model_suffix_for(generator)
        logger.info(f"Stages 1-3 retry a failed call up to {config.generator_max_retries} time(s) "
                    f"(generator.max_retries).")
    except Exception as e:
        logger.error(f"Generator Engine setup failed: {e}")
        sys.exit(1)
        
    # 3. Validate batch mode compatibility (early check before any stage)
    if args.batch:
        from pipeline.generator import BatchCapableGenerator
        if not isinstance(generator, BatchCapableGenerator):
            logger.error("Batch mode requires a generator engine that supports batch operations (Gemini or OpenAI).")
            sys.exit(1)

    if args.batch:
        logger.info("Batch mode enabled. All stages will use the Batch API with polling.")
        
    if args.single_pass:
        model_suffix = single_pass_suffix(model_suffix)

    # 5. Resolve the run plan: relations map, Stage 1 drafts and Stage 3 polish from cache, file or to be generated
    try:
        plan = resolve_run_plan(args, config, simanim_list, context_list, model_suffix)
    except PlanError as e:
        logger.error(str(e))
        sys.exit(1)
    if args.dry_run:
        for line in format_run_plan(plan, args, config, generator, model_suffix).split("\n"):
            logger.info(line)
        sys.exit(0)
    relations_text = plan.relations_text
    target_generation_simanim = plan.target_simanim
    drafts = dict(plan.drafts)
    missing_simanim = plan.missing_drafts

    # 6. Sefaria Extractor Setup
    try:
        extractor = SefariaExtractor(
            section_name=config.halachic_section,
            base_url=config.sefaria_base_url,
            timeout=config.sefaria_timeout,
            retries=config.sefaria_retries,
            ssl_verify=config.ssl_verify
        )
    except Exception as e:
        logger.error(f"Sefaria Extractor initialization failed: {e}")
        sys.exit(1)

    if args.single_pass:
        finish_run(run_single_pass_stages(target_generation_simanim, plan.polished, extractor, generator,
                                          config, args, model_suffix))
        return

    # 7. Generate the missing Stage 1 drafts (each saved as soon as it is ready; any failure aborts the run)
    if missing_simanim:
        logger.info(f"Generating missing Stage 1 drafts for Simanim: {missing_simanim}...")
        drafts.update(generate_drafts(missing_simanim, extractor, generator, config, args, model_suffix))

    if args.stage_1_only:
        logger.info("Stage 1 completed. Exiting as requested by --stage-1-only.")
        sys.exit(0)

    # 8. Generate relations map if it was not loaded
    if not relations_text:
        logger.info("Executing Stage 2: Cross-Siman Relations Analysis...")
        relations_name = relations_base_name(config, plan.context_simanim, model_suffix)
        relations_text = generate_relations(drafts, generator, config, args, relations_name)

        # Save relations map
        save_output_file(
            directory=config.relations_dir,
            base_name=relations_name,
            extension="txt",
            content=relations_text,
            logger=logger,
            label=f"{STAGE_RELATIONS} output for Simanim {plan.context_simanim}"
        )

    # 9. Stage 3+: polish and nikkud, dictionary nikkud, transcript, TTS (per-siman failures do not stop the run)
    finish_run(run_lesson_stages(target_generation_simanim, drafts, plan.polished, relations_text, generator,
                                 config, args, model_suffix))


def finish_run(summary: RunSummary) -> None:
    summary.log()
    if summary.has_failures:
        logger.error("Pipeline finished with failures.")
        sys.exit(1)
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except BatchTimeoutError as e:
        logger.error(str(e))
        sys.exit(1)
