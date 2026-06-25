import os
import argparse
import sys
from dotenv import load_dotenv

# Load environment variables from .env file at startup
load_dotenv()

from pipeline.config import PipelineConfig
from pipeline.input_parser import parse_simanim_string
from pipeline.extractor import SefariaExtractor
from pipeline.logger import get_logger
from pipeline.utils import save_output_file
from pipeline.factory import create_generator_engine, create_tts_engine

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
            extension="mp3",
            content=None,
            logger=logger,
            write_callback=tts_callback
        )
    except Exception as e:
        logger.error(f"TTS synthesis failed for Siman {siman}: {e}")

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
        help="Use OpenAI Batch API to submit the generation asynchronously."
    )
    parser.add_argument(
        "--retrieve-batch",
        type=str,
        default=None,
        help="Retrieve and save a previously submitted OpenAI batch generation draft using its Batch ID."
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
        
    # 3. Handle OpenAI Batch Retrieval (Stage 1 Draft retrieval)
    if args.retrieve_batch:
        if not config.openai_api_key:
            logger.error("No OpenAI API key found in config.yaml or OPENAI_API_KEY environment variable. Exiting.")
            sys.exit(1)
            
        from pipeline.generator import BatchCapableGenerator
        if not isinstance(generator, BatchCapableGenerator):
            logger.error("Batch retrieval is only supported for generator engines that support batch processing.")
            sys.exit(1)
            
        logger.info(f"Retrieving OpenAI Batch job: {args.retrieve_batch}...")
        try:
            result = generator.retrieve_batch_result(args.retrieve_batch)
            status = result.get("status")
            logger.info(f"Batch status: {status}")
            
            if status == "completed":
                script_text = result.get("content")
                siman = simanim_list[0] if simanim_list else 94
                
                save_output_file(
                    directory=config.drafts_dir,
                    base_name=f"{config.section_slug}_Siman_{siman}_{model_suffix}_draft",
                    extension="txt",
                    content=script_text,
                    logger=logger
                )
                
                logger.info("\n--- Batch Retrieval Complete ---")
                logger.info("The raw Stage 1 draft has been saved. Run the pipeline normally to complete Stage 2 and Stage 3.")
                
            elif status in ["validating", "in_progress"]:
                logger.info("The batch job is still running. Please try retrieving it again later.")
            else:
                logger.error(f"Batch job failed or was cancelled. Status: {status}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Failed to retrieve batch job: {e}")
            sys.exit(1)
        return
        
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
            # Batch mode: submit OpenAI batch jobs for all missing drafts and exit
            from pipeline.generator import BatchCapableGenerator
            if not isinstance(generator, BatchCapableGenerator):
                logger.error("Batch processing is only supported for generator engines that support batch processing.")
                sys.exit(1)
                
            logger.info("Submitting missing drafts as batch jobs...")
            for siman in missing_simanim:
                try:
                    logger.info(f"Compiling Sefaria context for Siman {siman}...")
                    micro_context = extractor.compile_simanim_context([siman], target_simanim=[siman])
                    
                    batch_input_path = os.path.join(config.cache_dir, f"openai_batch_siman_{siman}.jsonl")
                    batch_id = generator.create_batch_generation_job(
                        siman=siman,
                        master_context=micro_context,
                        system_instruction=config.gemini_system_instruction,
                        batch_input_path=batch_input_path,
                        commentators_desc=config.section_metadata["prompt_commentators_desc"]
                    )
                    logger.info(f"Submitted Batch Job for Siman {siman}. Batch ID: {batch_id}")
                    print(f"\n[BATCH_ID] {batch_id} (Siman {siman})\n")
                except Exception as e:
                    logger.error(f"Failed to submit batch job for Siman {siman}: {e}")
            logger.info("Batch submissions complete. Please run with --retrieve-batch <id> once complete.")
            return

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
        # Generate new relations map
        logger.info("Executing Stage 2: Cross-Siman Relations Analysis...")
        try:
            relations_text = generator.analyze_cross_relations(drafts, config.relations_instruction)
            
            # Save relations map immediately
            save_output_file(
                directory=config.relations_dir,
                base_name=f"{config.section_slug}_Relations_{range_str}_{model_suffix}",
                extension="txt",
                content=relations_text,
                logger=logger
            )
        except Exception as e:
            logger.error(f"Stage 2 Relations Analysis failed: {e}")
            sys.exit(1)

    # 9. Stage 3: Lesson Monologue styling & TTS
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
