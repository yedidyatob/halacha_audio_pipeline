import os
import argparse
import sys
import time
from dotenv import load_dotenv

# Load environment variables from .env file at startup
load_dotenv()

from pipeline.config import PipelineConfig
from pipeline.input_parser import parse_simanim_string
from pipeline.extractor import SefariaExtractor
from pipeline.logger import get_logger
from pipeline.utils import save_output_file
from pipeline.factory import create_generator_engine, create_tts_engine
from pipeline.gematria import int_to_gematria

logger = get_logger("halacha_pipeline_cli")

def process_and_save_outputs(siman: int, script_text: str, relations_text: str, generator, config, args, model_suffix: str):
    """
    Polishes the raw draft using the relations map, saves transcripts (both history copy and latest copy),
    and synthesizes the audio via TTS if not skipped.
    Consolidates the file saving and verification logic to keep the codebase DRY.
    """
    # Perform the Stage 3 polish pass (combining draft and relations)
    try:
        polished_text = generator.polish_siman_script(script_text, relations_text, config.polishing_instruction)
    except Exception as e:
        logger.error(f"Failed to polish script for Siman {siman}: {e}")
        raise
    
    # Save script transcript file
    save_output_file(
        directory=config.output_dir,
        base_name=f"{config.section_slug}_Siman_{siman}_{model_suffix}_transcript",
        extension="txt",
        content=polished_text,
        logger=logger
    )
    
    # Proceed to TTS if not skipped
    if args.skip_tts:
        logger.info("Skipping TTS synthesis as requested by --skip-tts.")
        return
        
    try:
        tts_engine = create_tts_engine(config)
        
        # Synthesize directly to the audio files using a write_callback
        def tts_callback(path: str) -> None:
            logger.info(f"Synthesizing script for Siman {siman} to history audio file: {path}...")
            tts_engine.synthesize(text=polished_text, output_path=path)
            
        save_output_file(
            directory=config.output_dir,
            base_name=f"{config.section_slug}_Siman_{siman}_{model_suffix}",
            extension=getattr(tts_engine, "file_extension", "mp3"),
            content=None,
            logger=logger,
            write_callback=tts_callback
        )
    except Exception as e:
        logger.error(f"TTS synthesis failed for Siman {siman}: {e}")

def poll_batch_jobs(generator, jobs: dict, poll_interval: int, timeout: int, stage_name: str, logger_ref=None):
    """
    Polls a dict of {label: job_id} until all complete or timeout.
    Raises RuntimeError on failure, TimeoutError on timeout.
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
    if failed:
        raise RuntimeError(f"{stage_name} batch jobs failed: {failed}")

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

    # Generate missing drafts
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
                    base_name=f"{config.section_slug}_Siman_{siman}_{model_suffix}_draft",
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
                        base_name=f"{config.section_slug}_Siman_{siman}_{model_suffix}_draft",
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
            base_name=f"{config.section_slug}_Relations_{range_str}_{model_suffix}",
            extension="txt",
            content=relations_text,
            logger=logger
        )

    # 9. Stage 3: Lesson Monologue styling & TTS
    if args.batch:
        # Batch mode: Submit all polishing jobs, poll, then save and run TTS
        stage3_jobs = {}
        for siman in target_generation_simanim:
            if siman not in drafts:
                logger.error(f"Cannot polish Siman {siman}: Stage 1 draft is missing.")
                continue
            user_prompt = generator._get_polishing_user_prompt(drafts[siman], relations_text)
            job_id = generator.submit_batch(
                system_instruction=config.polishing_instruction,
                user_prompt=user_prompt,
                temperature=0.1,
                custom_id=f"stage3_siman_{siman}",
                cache_dir=config.cache_dir
            )
            stage3_jobs[siman] = job_id
            logger.info(f"Submitted Stage 3 batch job for Siman {siman}. Job ID: {job_id}")

        if stage3_jobs:
            poll_batch_jobs(generator, stage3_jobs, args.batch_poll_interval, args.batch_timeout, "Stage 3 (Polishing)")

            for siman, job_id in stage3_jobs.items():
                polished_text = generator.get_batch_result(job_id)

                # Save polished transcript
                save_output_file(
                    directory=config.output_dir,
                    base_name=f"{config.section_slug}_Siman_{siman}_{model_suffix}_transcript",
                    extension="txt",
                    content=polished_text,
                    logger=logger
                )

                # Proceed to TTS if not skipped
                if not args.skip_tts:
                    try:
                        tts_engine = create_tts_engine(config)
                        def tts_callback(path: str, _text=polished_text, _siman=siman) -> None:
                            logger.info(f"Synthesizing script for Siman {_siman} to: {path}...")
                            tts_engine.synthesize(text=_text, output_path=path)
                        save_output_file(
                            directory=config.output_dir,
                            base_name=f"{config.section_slug}_Siman_{siman}_{model_suffix}",
                            extension=getattr(tts_engine, "file_extension", "mp3"),
                            content=None,
                            logger=logger,
                            write_callback=tts_callback
                        )
                    except Exception as e:
                        logger.error(f"TTS synthesis failed for Siman {siman}: {e}")
    else:
        # Synchronous mode
        for siman in target_generation_simanim:
            if siman not in drafts:
                logger.error(f"Cannot polish Siman {siman}: Stage 1 draft is missing from drafts database.")
                continue
                
            logger.info(f"\n--- Starting Stage 3 Monologue Generation for Siman {siman} ---")
            try:
                draft_text = drafts[siman]
                process_and_save_outputs(
                    siman=siman,
                    script_text=draft_text,
                    relations_text=relations_text,
                    generator=generator,
                    config=config,
                    args=args,
                    model_suffix=model_suffix
                )
            except Exception as e:
                logger.error(f"Failed Stage 3 Monologue Generation for Siman {siman}: {e}")
                sys.exit(1)

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
