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
from pipeline.gematria import int_to_gematria
from pipeline.apply_nikkud import apply_nikkud_to_abbreviations
from pipeline.pronunciation import create_pronunciation_vocalizer, format_changes
from pipeline.retry import call_with_retries

logger = get_logger("halacha_pipeline_cli")

STAGE_DRAFTS = "Stage 1 (Drafts)"
STAGE_RELATIONS = "Stage 2 (Relations)"
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


def _batch_first_attempts(batch: BatchCalls, items: dict, stage: str, args) -> dict:
    """
    Submits one batch job per item and polls them all. Returns {key: first_attempt}: fetching
    the job's result, or raising the submit error / job failure.
    """
    def failed(reason: str):
        def attempt():
            raise RuntimeError(reason)
        return attempt

    attempts, jobs = {}, {}
    for key, value in items.items():
        try:
            jobs[key] = batch.submit(key, value)
        except Exception as e:
            logger.error(f"{item_label(key)}: {stage} batch submit failed: {e}")
            attempts[key] = failed(f"batch submit failed: {e}")
    if jobs:
        statuses = poll_batch_jobs(batch.generator, jobs, args.batch_poll_interval, args.batch_timeout, stage)
        for key, job_id in jobs.items():
            if statuses[key] == "failed":
                attempts[key] = failed(f"batch job failed (job_id: {job_id})")
            else:
                attempts[key] = lambda key=key, job_id=job_id: batch.fetch(key, items[key], job_id)
    return attempts


def run_llm_stage(summary: RunSummary, items: dict, stage: str, call, max_retries: int, args,
                  batch: BatchCalls, on_result: Optional[Callable[[Any, Any], None]] = None) -> dict:
    """
    Runs an LLM stage over {key: value}, sync or batch (args.batch). Each item gets a first
    attempt - call(key, value), or in batch mode its batch job's result - and up to max_retries
    synchronous call(key, value) retries (pipeline.retry). on_result(key, result) runs as soon
    as an item succeeds (e.g. to save it). Returns {key: result} for the items that succeeded;
    the others are recorded in the summary with the last reason.
    """
    first_attempts = _batch_first_attempts(batch, items, stage, args) if args.batch else {}

    def attempt(key, value):
        result = call_with_retries(lambda: call(key, value), max_retries, f"{item_label(key)} {stage}",
                                   first_attempt=first_attempts.get(key))
        if on_result is not None:
            on_result(key, result)
        return result

    return run_per_siman(summary, items, stage, attempt)


def abort_if_failed(summary: RunSummary, stage: str, saved: dict) -> None:
    """Stages 1 and 2 are needed by every siman: any failure ends the run (after saving the rest)."""
    if not summary.has_failures:
        return
    summary.log()
    logger.error(f"{stage} failed; aborting the run. {len(saved)} result(s) were saved to the cache "
                 f"and are reused on a rerun.")
    sys.exit(1)


def save_draft(siman: int, draft_text: str, config, model_suffix: str) -> None:
    save_output_file(
        directory=config.drafts_dir,
        base_name=f"{config.section_slug}_Siman_{siman}_{model_suffix}_draft",
        extension="txt",
        content=draft_text,
        logger=logger
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
    commentators = config.section_metadata["prompt_commentators_desc"]
    drafts = run_llm_stage(
        summary, contexts, STAGE_DRAFTS,
        call=lambda siman, context: generator.generate_siman_script(
            siman=siman, master_context=context,
            system_instruction=config.gemini_system_instruction, commentators_desc=commentators
        ),
        max_retries=config.generator_max_retries, args=args,
        batch=BatchCalls(
            generator,
            submit=lambda siman, context: generator.submit_batch(
                system_instruction=config.gemini_system_instruction,
                user_prompt=generator._get_generation_user_prompt(int_to_gematria(siman), context, commentators),
                temperature=generator.temperature,
                custom_id=f"stage1_siman_{siman}",
                cache_dir=config.cache_dir
            ),
            fetch=lambda siman, context, job_id: generator.get_batch_result(job_id),
        ),
        on_result=lambda siman, draft: save_draft(siman, draft, config, model_suffix),
    )
    abort_if_failed(summary, STAGE_DRAFTS, drafts)
    return drafts


RELATIONS_KEY = "Relations map"


def generate_relations(drafts: dict, generator, config, args) -> str:
    """Stage 2: the cross-siman relations map (sync or batch, with generator.max_retries). Aborts on failure."""
    summary = RunSummary()
    result = run_llm_stage(
        summary, {RELATIONS_KEY: drafts}, STAGE_RELATIONS,
        call=lambda _, drafts: generator.analyze_cross_relations(drafts, config.relations_instruction),
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
    abort_if_failed(summary, STAGE_RELATIONS, result)
    return result[RELATIONS_KEY]


def polish_simanim(drafts: dict, relations_text: str, generator, config, args, summary: RunSummary) -> dict:
    """Stage 3: {siman: draft} -> {siman: polished_text} (sync or batch, with generator.max_retries)."""
    return run_llm_stage(
        summary, drafts, STAGE_POLISH,
        call=lambda siman, draft: generator.polish_siman_script(draft, relations_text, config.polishing_instruction),
        max_retries=config.generator_max_retries, args=args,
        batch=BatchCalls(
            generator,
            submit=lambda siman, draft: generator.submit_batch(
                system_instruction=config.polishing_instruction,
                user_prompt=generator._get_polishing_user_prompt(draft, relations_text),
                temperature=0.1,
                custom_id=f"stage3_siman_{siman}",
                cache_dir=config.cache_dir
            ),
            fetch=lambda siman, draft, job_id: generator.get_batch_result(job_id),
        ),
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
    return run_llm_stage(
        summary, texts, STAGE_PRONUNCIATION,
        call=lambda siman, text: vocalizer.vocalize_once(text),
        max_retries=vocalizer.max_retries, args=args,
        batch=BatchCalls(
            vocalizer.generator,
            submit=lambda siman, text: vocalizer.submit_batch(
                text, custom_id=f"stage3_5_siman_{siman}", cache_dir=config.cache_dir
            ),
            fetch=lambda siman, text, job_id: vocalizer.batch_result(text, job_id),
        ),
    )


def save_transcript(siman: int, text: str, changes, config, model_suffix: str) -> None:
    """
    Stage 3.6: saves the final transcript and, if the pronunciation stage ran, the list of
    words it vocalized (original TAB vocalized) for review.
    """
    base_name = f"{config.section_slug}_Siman_{siman}_{model_suffix}"
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
        base_name=f"{config.section_slug}_Siman_{siman}_{model_suffix}",
        extension=getattr(tts_engine, "file_extension", "mp3"),
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


def poll_batch_jobs(generator, jobs: dict, poll_interval: int, timeout: int, stage_name: str):
    """
    Polls a dict of {label: job_id} until all complete or timeout.
    Returns {label: "completed" | "failed"}. Raises TimeoutError on timeout.
    """
    start_time = time.time()
    pending = dict(jobs)
    results = {}

    logger.info(f"Polling {len(pending)} {stage_name} batch job(s) (interval: {poll_interval}s, timeout: {timeout}s)...")

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
                logger.info(f"\u2705 {stage_name} batch job completed: {label}")
                results[label] = status
                del pending[label]
            elif status == "failed":
                logger.error(f"\u274c {stage_name} batch job failed: {label} (job_id: {job_id})")
                results[label] = status
                del pending[label]

        if pending:
            elapsed_min = int((time.time() - start_time) / 60)
            logger.info(
                f"\u23f3 {stage_name}: {len(pending)} job(s) still pending, "
                f"{len(results)} completed. Elapsed: {elapsed_min}m. "
                f"Polling again in {poll_interval}s..."
            )
            time.sleep(poll_interval)

    failed = [label for label, status in results.items() if status == "failed"]
    if failed:
        logger.error(f"{stage_name}: {len(failed)} of {len(results)} batch job(s) failed: {failed}")
    else:
        logger.info(f"{stage_name}: All {len(results)} batch job(s) completed successfully.")
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
        
    # 5. Stage 2: Cross-Siman Relations Analysis - Resolve Relations Map First
    relations_text = ""
    relations_path = None
    range_str = "_".join(str(s) for s in context_list)
    
    if args.relations_file:
        if not os.path.exists(args.relations_file):
            logger.error(f"Relations file not found at: {args.relations_file}")
            sys.exit(1)
        try:
            with open(args.relations_file, "r", encoding="utf-8") as f:
                relations_text = f.read()
            logger.info(f"Loaded pre-existing relations map: {args.relations_file} ({len(relations_text)} chars).")
        except Exception as e:
            logger.error(f"Failed to read relations file: {e}")
            sys.exit(1)
    else:
        # Determine cached relations filename based on context range
        relations_filename = f"{config.section_slug}_Relations_{range_str}_{model_suffix}.txt"
        relations_path = os.path.join(config.relations_dir, relations_filename)
        
        if os.path.exists(relations_path) and not args.overwrite_cache:
            try:
                with open(relations_path, "r", encoding="utf-8") as f:
                    relations_text = f.read()
                logger.info(f"Loaded cached Stage 2 relations map: {relations_path} ({len(relations_text)} chars).")
            except Exception as e:
                logger.warning(f"Failed to read cached relations: {e}. Will regenerate.")

    # Apply debug limit if specified
    target_generation_simanim = simanim_list
    if args.limit_lessons is not None:
        if args.limit_lessons <= 0:
            logger.error("Limit lessons count must be greater than zero.")
            sys.exit(1)
        target_generation_simanim = simanim_list[:args.limit_lessons]
        logger.info(f"[DEBUG LIMIT] Restricting Stage 3 styling to: {target_generation_simanim}")

    # Determine which drafts we actually need to load/generate
    if relations_text:
        # We already have relations, so we only need drafts for the target Simanim we are polishing
        drafts_needed = target_generation_simanim
        logger.info(f"Relations map is already loaded. Resolving drafts only for target Simanim: {drafts_needed}")
    else:
        # We need to generate relations, so we need drafts for all Simanim in context list
        drafts_needed = context_list
        logger.info(f"Relations map needs to be generated. Resolving drafts for entire context range: {drafts_needed}")

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

    # 7. Resolve Stage 1 Drafts for drafts_needed
    drafts = {}
    missing_simanim = []
    
    # Check what drafts already exist in cache
    for siman in drafts_needed:
        draft_filename = f"{config.section_slug}_Siman_{siman}_{model_suffix}_draft.txt"
        draft_path = os.path.join(config.drafts_dir, draft_filename)
        
        if os.path.exists(draft_path) and not args.overwrite_cache:
            try:
                with open(draft_path, "r", encoding="utf-8") as f:
                    drafts[siman] = f.read()
                logger.info(f"Loaded cached Stage 1 draft for Siman {siman} ({len(drafts[siman])} chars).")
            except Exception as e:
                logger.warning(f"Failed to read cached draft at '{draft_path}': {e}. Will regenerate.")
                missing_simanim.append(siman)
        else:
            missing_simanim.append(siman)

    # Generate missing drafts (each saved as soon as it is ready; any failure aborts the run)
    if missing_simanim:
        logger.info(f"Generating missing Stage 1 drafts for Simanim: {missing_simanim}...")
        drafts.update(generate_drafts(missing_simanim, extractor, generator, config, args, model_suffix))

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
        relations_text = generate_relations(drafts, generator, config, args)

        # Save relations map
        save_output_file(
            directory=config.relations_dir,
            base_name=f"{config.section_slug}_Relations_{range_str}_{model_suffix}",
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
