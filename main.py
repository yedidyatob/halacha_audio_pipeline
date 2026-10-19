import os
import argparse
import sys
import time
from dataclasses import dataclass
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
from pipeline.gematria import int_to_gematria
from pipeline.apply_nikkud import apply_nikkud_to_abbreviations
from pipeline.pronunciation import create_pronunciation_vocalizer, format_changes

logger = get_logger("halacha_pipeline_cli")

STAGE_POLISH = "Stage 3 (Polishing)"
STAGE_NIKKUD = "Stage 3.4 (Dictionary nikkud)"
STAGE_PRONUNCIATION = "Stage 3.5 (Pronunciation nikkud)"
STAGE_TRANSCRIPT = "Stage 3.6 (Transcript save)"
STAGE_TTS = "TTS"


class SimanStageError(Exception):
    """A single siman failed at a named stage; the run continues with the next siman."""

    def __init__(self, stage: str, reason: str):
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


def run_siman_stage(siman: int, stage: str, fn):
    """Runs fn(); any exception is logged and re-raised as SimanStageError(stage, reason)."""
    try:
        return fn()
    except SimanStageError:
        raise
    except Exception as e:
        logger.error(f"Siman {siman} failed at {stage}: {e}")
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
            lines.append(f"  - Siman {siman} | stage: {stage} | reason: {reason}")
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


def run_batch_stage(batch_generator, summary: RunSummary, items: dict, stage: str, submit, fetch, args) -> dict:
    """
    Batch counterpart of run_per_siman: submit(siman, value) -> job_id for every item,
    poll all jobs, then fetch(siman, value, job_id) -> result for each completed job.
    Submit errors, failed jobs and fetch errors fail only their siman.
    """
    jobs = run_per_siman(summary, items, stage, submit)
    if not jobs:
        return {}
    statuses = poll_batch_jobs(batch_generator, jobs, args.batch_poll_interval, args.batch_timeout,
                               stage, raise_on_failure=False)
    completed = {}
    for siman, job_id in jobs.items():
        if statuses[siman] == "failed":
            summary.failure(siman, stage, f"batch job failed (job_id: {job_id})")
        else:
            completed[siman] = (items[siman], job_id)
    return run_per_siman(summary, completed, stage, lambda siman, item: fetch(siman, *item))


def polish_simanim(drafts: dict, relations_text: str, generator, config, args, summary: RunSummary) -> dict:
    """Stage 3: {siman: draft} -> {siman: polished_text}."""
    if args.batch:
        return run_batch_stage(
            generator, summary, drafts, STAGE_POLISH,
            submit=lambda siman, draft: generator.submit_batch(
                system_instruction=config.polishing_instruction,
                user_prompt=generator._get_polishing_user_prompt(draft, relations_text),
                temperature=0.1,
                custom_id=f"stage3_siman_{siman}",
                cache_dir=config.cache_dir
            ),
            fetch=lambda siman, draft, job_id: generator.get_batch_result(job_id),
            args=args,
        )
    return run_per_siman(
        summary, drafts, STAGE_POLISH,
        lambda siman, draft: generator.polish_siman_script(draft, relations_text, config.polishing_instruction)
    )


def apply_dictionary_nikkud(polished: dict, summary: RunSummary) -> dict:
    """Stage 3.4: {siman: polished_text} -> {siman: text} with the dictionary nikkud applied."""
    return run_per_siman(summary, polished, STAGE_NIKKUD, lambda siman, text: apply_nikkud_to_abbreviations(text))


def vocalize_simanim(texts: dict, vocalizer, config, args, summary: RunSummary) -> dict:
    """
    Stage 3.5: {siman: text} -> {siman: (text, changes)}. changes is None when the
    pronunciation stage is disabled (vocalizer is None) and the text passes through.
    """
    if vocalizer is None:
        return {siman: (text, None) for siman, text in texts.items()}
    if args.batch:
        return run_batch_stage(
            vocalizer.generator, summary, texts, STAGE_PRONUNCIATION,
            submit=lambda siman, text: vocalizer.submit_batch(
                text, custom_id=f"stage3_5_siman_{siman}", cache_dir=config.cache_dir
            ),
            fetch=lambda siman, text, job_id: vocalizer.batch_result(text, job_id),
            args=args,
        )
    return run_per_siman(summary, texts, STAGE_PRONUNCIATION, lambda siman, text: vocalizer.vocalize(text))


AUDIO_EXTENSION = "mp3"


def lesson_base_name(config, siman: int, model_suffix: str) -> str:
    """Base name of a siman's outputs in output_dir (transcript, pronunciation changes, audio)."""
    return f"{config.section_slug}_Siman_{siman}_{model_suffix}"


def draft_base_name(config, siman: int, model_suffix: str) -> str:
    return f"{config.section_slug}_Siman_{siman}_{model_suffix}_draft"


def relations_base_name(config, context_simanim: list, model_suffix: str) -> str:
    range_str = "_".join(str(s) for s in context_simanim)
    return f"{config.section_slug}_Relations_{range_str}_{model_suffix}"


class PlanError(ValueError):
    """The run cannot start (e.g. --relations-file not found); the message says why."""


@dataclass
class RunPlan:
    """What a run will load from cache and what it will generate; resolved before any API call."""
    context_simanim: list
    target_simanim: list          # Stage 3+ (after --limit-lessons)
    relations_source: str         # "file" (--relations-file), "cache" or "generate"
    relations_path: str           # file it is loaded from, or where the generated map is saved
    relations_text: str           # "" when it will be generated
    draft_paths: dict             # {siman: cache path} for every draft needed
    drafts: dict                  # {siman: text} loaded from cache
    missing_drafts: list          # simanim whose Stage 1 draft will be generated

    @property
    def drafts_needed(self) -> list:
        return list(self.draft_paths)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def resolve_run_plan(args, config, simanim_list: list, context_list: list, model_suffix: str) -> RunPlan:
    """
    Resolves the relations map (--relations-file, cache unless --overwrite-cache, or generate)
    and the Stage 1 drafts it needs (cached vs to generate), reading only local files.
    Raises PlanError if the run cannot start.
    """
    relations_text, relations_source = "", "generate"
    if args.relations_file:
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

    if relations_text:
        # relations are available, so only the target simanim being polished need drafts
        drafts_needed = target_simanim
        logger.info(f"Relations map is already loaded. Resolving drafts only for target Simanim: {drafts_needed}")
    else:
        # generating relations needs a draft for every siman in the context list
        drafts_needed = context_list
        logger.info(f"Relations map needs to be generated. Resolving drafts for entire context range: {drafts_needed}")

    draft_paths, drafts, missing = {}, {}, []
    for siman in drafts_needed:
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

    return RunPlan(context_simanim=context_list, target_simanim=target_simanim,
                   relations_source=relations_source, relations_path=relations_path,
                   relations_text=relations_text, draft_paths=draft_paths, drafts=drafts,
                   missing_drafts=missing)


def format_run_plan(plan: RunPlan, args, config, generator, vocalizer, model_suffix: str) -> str:
    """The --dry-run report: what a real run with these arguments would load, generate and write."""
    calls = "batch jobs" if args.batch else "synchronous calls"
    lines = [
        "===== Dry run: plan (no API / LLM / TTS / Sefaria calls were made) =====",
        f"Mode: {'batch' if args.batch else 'sync'} | generator: {config.generator_engine} {generator.model_name}"
        + (" | --overwrite-cache" if args.overwrite_cache else ""),
        f"Target simanim (Stage 3+): {plan.target_simanim}"
        + (f" (--limit-lessons {args.limit_lessons})" if args.limit_lessons is not None else ""),
        f"Context simanim: {plan.context_simanim}",
    ]
    if plan.relations_source == "file":
        lines.append(f"Stage 2 relations map: loaded from --relations-file {plan.relations_path}")
    elif plan.relations_source == "cache":
        lines.append(f"Stage 2 relations map: loaded from cache {plan.relations_path}")
    elif args.stage_1_only:
        lines.append("Stage 2 relations map: not cached; not generated (--stage-1-only)")
    else:
        lines.append(f"Stage 2 relations map: will be GENERATED ({'a batch job' if args.batch else 'a synchronous call'}) "
                     f"from the drafts of {plan.drafts_needed} and saved to {plan.relations_path}")
    cached = [s for s in plan.drafts_needed if s in plan.drafts]
    lines.append(f"Stage 1 drafts needed: {plan.drafts_needed}")
    lines.append(f"  from cache ({len(cached)}): {cached or 'none'}")
    lines += [f"    Siman {s}: {plan.draft_paths[s]}" for s in cached]
    lines.append(f"  to be GENERATED ({len(plan.missing_drafts)}, as {calls}, after a Sefaria fetch each): "
                 f"{plan.missing_drafts or 'none'}")
    lines += [f"    Siman {s} -> {plan.draft_paths[s]}" for s in plan.missing_drafts]
    if args.stage_1_only:
        lines.append("--stage-1-only: the run stops after Stage 1 (and the heuristic evaluation).")
        return "\n".join(lines)
    pronunciation = (f"{config.pronunciation_engine} {vocalizer.generator.model_name}" if vocalizer is not None
                     else "disabled (pronunciation.enabled: false)")
    lines += [
        f"Stage 3 polish ({calls}) -> 3.4 dictionary nikkud -> 3.5 pronunciation nikkud: {plan.target_simanim}",
        f"  Stage 3.5 pronunciation: {pronunciation}",
        f"Transcripts / TTS (tts.engine {config.tts_engine}{', skipped: --skip-tts' if args.skip_tts else ''}):",
    ]
    for siman in plan.target_simanim:
        base = os.path.join(config.output_dir, lesson_base_name(config, siman, model_suffix))
        lines.append(f"    Siman {siman}: {base}_transcript.txt"
                     + ("" if args.skip_tts else f", {base}.{AUDIO_EXTENSION}"))
    lines.append("(Each saved file also gets a timestamped history copy.)")
    return "\n".join(lines)


def save_transcript(siman: int, text: str, changes, config, model_suffix: str) -> None:
    """
    Stage 3.6: saves the final transcript and, if the pronunciation stage ran, the list of
    words it vocalized (original TAB vocalized) for review.
    """
    base_name = lesson_base_name(config, siman, model_suffix)
    if changes is not None:
        save_output_file(
            directory=config.output_dir,
            base_name=f"{base_name}_pronunciation_changes",
            extension="txt",
            content=format_changes(changes),
            logger=logger
        )
    save_output_file(
        directory=config.output_dir,
        base_name=f"{base_name}_transcript",
        extension="txt",
        content=text,
        logger=logger
    )


def synthesize_siman_audio(siman: int, text: str, config, model_suffix: str) -> None:
    """TTS for one siman."""
    tts_engine = create_tts_engine(config)

    # Synthesize directly to the audio files using a write_callback
    def tts_callback(path: str) -> None:
        logger.info(f"Synthesizing script for Siman {siman} to: {path}...")
        tts_engine.synthesize(text=text, output_path=path)

    save_output_file(
        directory=config.output_dir,
        base_name=lesson_base_name(config, siman, model_suffix),
        extension=AUDIO_EXTENSION,
        content=None,
        logger=logger,
        write_callback=tts_callback
    )


def finish_siman(siman: int, text: str, changes, config, args, model_suffix: str) -> None:
    """Stage 3.6 (transcript save) then TTS unless --skip-tts. Raises SimanStageError on failure."""
    run_siman_stage(siman, STAGE_TRANSCRIPT, lambda: save_transcript(siman, text, changes, config, model_suffix))
    if args.skip_tts:
        logger.info("Skipping TTS synthesis as requested by --skip-tts.")
        return
    run_siman_stage(siman, STAGE_TTS, lambda: synthesize_siman_audio(siman, text, config, model_suffix))


def run_lesson_stages(target_simanim: list, drafts: dict, relations_text: str, generator, vocalizer,
                      config, args, model_suffix: str) -> RunSummary:
    """
    Stage 3 onwards for every target siman, one stage at a time (sync or batch):
    polish -> dictionary nikkud -> pronunciation nikkud (LLM) -> transcript save -> TTS.
    A failure removes only that siman from the later stages.
    """
    summary = RunSummary()
    available = {}
    for siman in target_simanim:
        if siman in drafts:
            available[siman] = drafts[siman]
        else:
            logger.error(f"Cannot polish Siman {siman}: Stage 1 draft is missing.")
            summary.failure(siman, STAGE_POLISH, "Stage 1 draft is missing")

    polished = polish_simanim(available, relations_text, generator, config, args, summary)
    vocalized = vocalize_simanim(apply_dictionary_nikkud(polished, summary), vocalizer, config, args, summary)
    finished = run_per_siman(
        summary, vocalized, STAGE_TRANSCRIPT,
        lambda siman, item: finish_siman(siman, *item, config, args, model_suffix)
    )
    for siman in finished:
        summary.success(siman)
    return summary


def poll_batch_jobs(generator, jobs: dict, poll_interval: int, timeout: int, stage_name: str, logger_ref=None,
                    raise_on_failure: bool = True):
    """
    Polls a dict of {label: job_id} until all complete or timeout.
    Returns {label: "completed" | "failed"}. Raises TimeoutError on timeout, and
    RuntimeError on any failed job when raise_on_failure is True.
    """
    log = logger_ref or logger
    start_time = time.time()
    pending = dict(jobs)
    results = {}

    log.info(f"Polling {len(pending)} {stage_name} batch job(s) (interval: {poll_interval}s, timeout: {timeout}s)...")

    while pending:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(
                f"Batch {stage_name} timed out after {int(elapsed)}s. "
                f"Still pending: {list(pending.keys())}"
            )

        for label, job_id in list(pending.items()):
            status = generator.get_batch_status(job_id)
            if status == "completed":
                log.info(f"\u2705 {stage_name} batch job completed: {label}")
                results[label] = status
                del pending[label]
            elif status == "failed":
                log.error(f"\u274c {stage_name} batch job failed: {label} (job_id: {job_id})")
                results[label] = status
                del pending[label]

        if pending:
            elapsed_min = int((time.time() - start_time) / 60)
            log.info(
                f"\u23f3 {stage_name}: {len(pending)} job(s) still pending, "
                f"{len(results)} completed. Elapsed: {elapsed_min}m. "
                f"Polling again in {poll_interval}s..."
            )
            time.sleep(poll_interval)

    # Check for any failures
    failed = [label for label, status in results.items() if status == "failed"]
    if failed and raise_on_failure:
        raise RuntimeError(f"{stage_name} batch jobs failed: {failed}")

    if failed:
        log.error(f"{stage_name}: {len(failed)} of {len(results)} batch job(s) failed: {failed}")
    else:
        log.info(f"{stage_name}: All {len(results)} batch job(s) completed successfully.")
    return results

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
        help="Force regeneration of Stage 1 drafts and Stage 2 relations maps, overwriting any cached files."
    )
    parser.add_argument(
        "--relations-file",
        type=str,
        default=None,
        help="Path to a pre-existing cross-relations map file. If specified, Stage 2 relations analysis is skipped."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan (relations map and drafts loaded from cache vs generated, simanim going through "
             "Stage 3/3.5/TTS, file paths) and exit without any API, LLM, TTS or Sefaria call."
    )
    parser.add_argument(
        "--stage-1-only",
        action="store_true",
        help="Only run Stage 1 (Data Extraction) and exit."
    )
    return parser.parse_args()

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
        model_suffix = generator.model_name.replace('.', '_')
    except Exception as e:
        logger.error(f"Generator Engine setup failed: {e}")
        sys.exit(1)
        
    # 3. Validate batch mode compatibility (early check before any stage)
    if args.batch:
        from pipeline.generator import BatchCapableGenerator
        if not isinstance(generator, BatchCapableGenerator):
            logger.error("Batch mode requires a generator engine that supports batch operations (Gemini or OpenAI).")
            sys.exit(1)

    # 4. Pronunciation stage (Stage 3.5) with its own engine/model from config.yaml
    try:
        vocalizer = create_pronunciation_vocalizer(config) if config.pronunciation_enabled else None
    except Exception as e:
        logger.error(f"Pronunciation stage setup failed: {e}")
        sys.exit(1)
    if vocalizer is not None:
        if args.batch and not vocalizer.supports_batch:
            logger.error(
                f"Batch mode requires a pronunciation engine that supports batch operations; "
                f"'{config.pronunciation_engine}' does not (pronunciation.engine in config.yaml)."
            )
            sys.exit(1)
    else:
        logger.info("Pronunciation stage disabled (pronunciation.enabled: false); skipping Stage 3.5.")
    if args.batch:
        logger.info("Batch mode enabled. All stages will use the Batch API with polling.")
        
    # 5. Resolve the run plan: relations map and Stage 1 drafts from cache, file or to be generated
    try:
        plan = resolve_run_plan(args, config, simanim_list, context_list, model_suffix)
    except PlanError as e:
        logger.error(str(e))
        sys.exit(1)
    if args.dry_run:
        for line in format_run_plan(plan, args, config, generator, vocalizer, model_suffix).split("\n"):
            logger.info(line)
        sys.exit(0)
    relations_text = plan.relations_text
    target_generation_simanim = plan.target_simanim
    drafts_needed = plan.drafts_needed
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

    # 7. Generate the missing Stage 1 drafts
    if missing_simanim:
        logger.info(f"Generating missing Stage 1 drafts for Simanim: {missing_simanim}...")
        
        if args.batch:
            # Batch mode: submit batch jobs for all missing drafts, poll until complete
            logger.info("Submitting Stage 1 batch jobs for missing drafts...")
            stage1_jobs = {}
            for siman in missing_simanim:
                try:
                    logger.info(f"Compiling Sefaria context for Siman {siman}...")
                    micro_context = extractor.compile_simanim_context([siman], target_simanim=[siman])
                    gematria_siman = int_to_gematria(siman)
                    user_prompt = generator._get_generation_user_prompt(
                        gematria_siman, micro_context, config.section_metadata["prompt_commentators_desc"]
                    )
                    job_id = generator.submit_batch(
                        system_instruction=config.gemini_system_instruction,
                        user_prompt=user_prompt,
                        temperature=generator.temperature,
                        custom_id=f"stage1_siman_{siman}",
                        cache_dir=config.cache_dir
                    )
                    stage1_jobs[siman] = job_id
                    logger.info(f"Submitted Stage 1 batch job for Siman {siman}. Job ID: {job_id}")
                except Exception as e:
                    logger.error(f"Failed to submit Stage 1 batch job for Siman {siman}: {e}")
                    sys.exit(1)

            # Poll until all Stage 1 jobs complete
            poll_batch_jobs(generator, stage1_jobs, args.batch_poll_interval, args.batch_timeout, "Stage 1 (Drafts)")

            # Retrieve and save Stage 1 results
            for siman, job_id in stage1_jobs.items():
                draft_text = generator.get_batch_result(job_id)
                drafts[siman] = draft_text
                save_output_file(
                    directory=config.drafts_dir,
                    base_name=draft_base_name(config, siman, model_suffix),
                    extension="txt",
                    content=draft_text,
                    logger=logger
                )
        else:
            # Normal synchronous mode
            for siman in missing_simanim:
                try:
                    logger.info(f"Compiling Sefaria context for Siman {siman}...")
                    micro_context = extractor.compile_simanim_context([siman], target_simanim=[siman])
                    
                    draft_text = generator.generate_siman_script(
                        siman=siman,
                        master_context=micro_context,
                        system_instruction=config.gemini_system_instruction,
                        commentators_desc=config.section_metadata["prompt_commentators_desc"]
                    )
                    drafts[siman] = draft_text
                    
                    # Save draft immediately
                    save_output_file(
                        directory=config.drafts_dir,
                        base_name=draft_base_name(config, siman, model_suffix),
                        extension="txt",
                        content=draft_text,
                        logger=logger
                    )
                except Exception as e:
                    logger.error(f"Failed to extract Stage 1 draft for Siman {siman}: {e}")
                    sys.exit(1)

    # 7.5 Run Heuristic Evaluation for resolved drafts
    logger.info("Executing Heuristic Evaluation on resolved drafts...")
    from pipeline.evaluator import evaluate_draft
    
    for siman in drafts_needed:
        if siman in drafts:
            logger.info(f"Evaluating draft quality for Siman {siman}...")
            try:
                eval_res = evaluate_draft(siman, drafts[siman], config_path=args.config)
                if not eval_res["success"]:
                    logger.warning(
                        f"\n"
                        f"============================================================\n"
                        f"⚠️⚠️⚠️ HEURISTIC EVALUATION WARNING FOR SIMAN {siman} ⚠️⚠️⚠️\n"
                        f"============================================================\n"
                        f"The generated Stage 1 draft failed heuristic coverage checks.\n\n"
                        f"{eval_res['report']}\n"
                        f"============================================================\n"
                    )
                else:
                    logger.info(f"Heuristic coverage evaluation for Siman {siman} PASSED.")
            except Exception as e:
                logger.error(f"Failed to run heuristic evaluation for Siman {siman}: {e}")

    if args.stage_1_only:
        logger.info("Stage 1 completed. Exiting as requested by --stage-1-only.")
        sys.exit(0)

    # 8. Generate relations map if it was not loaded
    if not relations_text:
        logger.info("Executing Stage 2: Cross-Siman Relations Analysis...")
        if args.batch:
            # Batch mode: submit relations analysis as batch job
            user_prompt = generator._get_relations_user_prompt(drafts)
            job_id = generator.submit_batch(
                system_instruction=config.relations_instruction,
                user_prompt=user_prompt,
                temperature=generator.temperature,
                custom_id="stage2_relations",
                cache_dir=config.cache_dir
            )
            logger.info(f"Submitted Stage 2 batch job. Job ID: {job_id}")
            poll_batch_jobs(generator, {"relations": job_id}, args.batch_poll_interval, args.batch_timeout, "Stage 2 (Relations)")
            relations_text = generator.get_batch_result(job_id)
        else:
            # Synchronous mode
            try:
                relations_text = generator.analyze_cross_relations(drafts, config.relations_instruction)
            except Exception as e:
                logger.error(f"Stage 2 Relations Analysis failed: {e}")
                sys.exit(1)

        # Save relations map
        save_output_file(
            directory=config.relations_dir,
            base_name=relations_base_name(config, plan.context_simanim, model_suffix),
            extension="txt",
            content=relations_text,
            logger=logger
        )

    # 9. Stage 3+: polish, dictionary nikkud, pronunciation nikkud, transcript, TTS (per-siman failures do not stop the run)
    summary = run_lesson_stages(target_generation_simanim, drafts, relations_text, generator, vocalizer,
                                config, args, model_suffix)

    summary.log()
    if summary.has_failures:
        logger.error("Pipeline finished with failures.")
        sys.exit(1)
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
