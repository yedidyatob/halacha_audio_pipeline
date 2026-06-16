import os
import argparse
import sys
from pipeline.config import PipelineConfig
from pipeline.input_parser import parse_simanim_string
from pipeline.extractor import SefariaExtractor
from pipeline.logger import get_logger

logger = get_logger("halacha_pipeline_cli")

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
        help="Use OpenAI Batch API to submit the generation asynchronously (saves 50% cost and bypasses rate limits)."
    )
    parser.add_argument(
        "--retrieve-batch",
        type=str,
        default=None,
        help="Retrieve and complete a previously submitted OpenAI batch generation job using its Batch ID."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 1. Parse Simanim list
    try:
        simanim_list = parse_simanim_string(args.simanim)
        logger.info(f"Target Simanim parsed: {simanim_list}")
        
        # Determine context list
        if args.context_range:
            context_list = parse_simanim_string(args.context_range)
            # Ensure target Simanim are included in the context list for completeness
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
        
    # If retrieving batch, bypass Sefaria extraction and normal generation steps
    if args.retrieve_batch:
        if not config.openai_api_key:
            logger.error("No OpenAI API key found in config.yaml or OPENAI_API_KEY environment variable. Exiting.")
            sys.exit(1)
            
        from pipeline.generator import OpenAIScriptGenerator
        generator = OpenAIScriptGenerator(
            api_key=config.openai_api_key,
            model_name=config.openai_model_name,
            temperature=config.openai_temperature,
            service_tier=config.openai_service_tier
        )
        
        logger.info(f"Retrieving OpenAI Batch job: {args.retrieve_batch}...")
        try:
            result = generator.retrieve_batch_result(args.retrieve_batch)
            status = result.get("status")
            logger.info(f"Batch status: {status}")
            
            if status == "completed":
                script_text = result.get("content")
                siman = simanim_list[0] if simanim_list else 94
                    
                # Perform the polish pass synchronously (draft is small enough to stay under TPM)
                try:
                    script_text = generator.polish_siman_script(script_text)
                except Exception as e:
                    logger.warning(f"Failed to polish script for Siman {siman}: {e}. Using raw draft script.")
                
                # Save script transcript file (latest copy and timestamped history copy)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                transcript_filename = f"Yoreh_Deah_Siman_{siman}_transcript.txt"
                transcript_path = os.path.join(config.output_dir, transcript_filename)
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(script_text)
                
                history_filename = f"Yoreh_Deah_Siman_{siman}_transcript_{timestamp}.txt"
                history_path = os.path.join(config.output_dir, history_filename)
                with open(history_path, "w", encoding="utf-8") as f:
                    f.write(script_text)
                logger.info(f"Saved polished transcript to: {transcript_path} and history copy: {history_path}")
                
                # Proceed to TTS if not skipped
                if args.skip_tts:
                    logger.info("Skipping TTS synthesis as requested by --skip-tts.")
                    return
                    
                tts_engine = config.get_tts_engine()
                
                # Synthesize directly to the timestamped history audio file
                audio_history_filename = f"Yoreh_Deah_Siman_{siman}_{timestamp}.mp3"
                audio_history_path = os.path.join(config.output_dir, audio_history_filename)
                logger.info(f"Synthesizing script for Siman {siman} to history audio file...")
                tts_engine.synthesize(text=script_text, output_path=audio_history_path)
                logger.info(f"Generated history MP3 saved: {audio_history_path}")
                
                # Attempt to save a copy as the main audio file (Yoreh_Deah_Siman_X.mp3)
                audio_filename = f"Yoreh_Deah_Siman_{siman}.mp3"
                audio_path = os.path.join(config.output_dir, audio_filename)
                try:
                    import shutil
                    shutil.copyfile(audio_history_path, audio_path)
                    logger.info(f"Updated latest MP3 copy at: {audio_path}")
                except Exception as copy_err:
                    logger.warning(
                        f"Could not overwrite latest MP3 copy at '{audio_path}' (it may be locked by a media player): {copy_err}. "
                        f"The generated audio remains fully saved in: '{audio_history_path}'."
                    )
                
            elif status in ["validating", "in_progress"]:
                logger.info("The batch job is still running. Please try retrieving it again later.")
            else:
                logger.error(f"Batch job failed or was cancelled. Status: {status}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Failed to retrieve batch job: {e}")
            sys.exit(1)
        return
        
    # 3. Sefaria Extraction (v3 text query)
    try:
        extractor = SefariaExtractor(
            base_url=config.sefaria_base_url,
            timeout=config.sefaria_timeout,
            retries=config.sefaria_retries
        )
        
        # Compile consolidated context for all requested Simanim
        logger.info(f"Compiling consolidated Sefaria context for Simanim: {context_list}...")
        master_context = extractor.compile_simanim_context(context_list)
        
        # Save master context to cache
        context_cache_path = os.path.join(config.cache_dir, "master_context.txt")
        with open(context_cache_path, "w", encoding="utf-8") as f:
            f.write(master_context)
        logger.info(f"Consolidated master context saved to cache: {context_cache_path}")
        
    except Exception as e:
        logger.error(f"Sefaria Extraction Failed: {e}")
        sys.exit(1)

    # 4. Script Generation
    try:
        # Check API key configuration for the selected generator engine
        if config.generator_engine == "gemini" and not config.gemini_api_key:
            logger.error("No Gemini API key found in config.yaml or GEMINI_API_KEY environment variable. Exiting.")
            sys.exit(1)
        elif config.generator_engine == "openai" and not config.openai_api_key:
            logger.error("No OpenAI API key found in config.yaml or OPENAI_API_KEY environment variable. Exiting.")
            sys.exit(1)
            
        generator = config.get_generator_engine()
        
        # Apply debug limit if specified
        target_generation_simanim = simanim_list
        if args.limit_lessons is not None:
            if args.limit_lessons <= 0:
                logger.error("Limit lessons count must be greater than zero.")
                sys.exit(1)
            target_generation_simanim = simanim_list[:args.limit_lessons]
            logger.info(
                f"[DEBUG LIMIT] Restricting generation to the first {args.limit_lessons} Siman(im): "
                f"{target_generation_simanim} (Master context size remains at {len(context_list)} Simanim)."
            )
            
        # Keep track of successful generation scripts for the TTS step
        scripts_to_synthesize = {}
        
        for siman in target_generation_simanim:
            try:
                if args.batch:
                    from pipeline.generator import OpenAIScriptGenerator
                    if not isinstance(generator, OpenAIScriptGenerator):
                        logger.error("Batch processing is only supported for the OpenAI generator engine.")
                        sys.exit(1)
                        
                    batch_input_path = os.path.join(config.cache_dir, f"openai_batch_siman_{siman}.jsonl")
                    batch_id = generator.create_batch_generation_job(
                        siman=siman,
                        master_context=master_context,
                        system_instruction=config.gemini_system_instruction,
                        batch_input_path=batch_input_path
                    )
                    logger.info("\n--- OpenAI Batch Job Submitted Successfully ---")
                    logger.info(f"Batch ID: {batch_id}")
                    logger.info(f"Saved local batch payload definition: {batch_input_path}")
                    logger.info("The job will complete in the background (typically within a few minutes).")
                    logger.info(f"To check status or retrieve the result, run the following command once completed:")
                    logger.info(f"  ..\\v\\Scripts\\python.exe main.py {siman} --retrieve-batch {batch_id} --skip-tts")
                    print(f"\n[BATCH_ID] {batch_id}\n")
                    continue
                    
                # Generate script using Master Reference Strategy
                script_text = generator.generate_siman_script(
                    siman=siman,
                    master_context=master_context,
                    system_instruction=config.gemini_system_instruction
                )
                
                # Perform a secondary TTS verification and polish pass
                try:
                    script_text = generator.polish_siman_script(script_text)
                except Exception as e:
                    logger.warning(f"Failed to polish script for Siman {siman}: {e}. Falling back to draft script.")
                
                # Save script transcript file (latest copy and timestamped history copy)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                transcript_filename = f"Yoreh_Deah_Siman_{siman}_transcript.txt"
                transcript_path = os.path.join(config.output_dir, transcript_filename)
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(script_text)
                
                history_filename = f"Yoreh_Deah_Siman_{siman}_transcript_{timestamp}.txt"
                history_path = os.path.join(config.output_dir, history_filename)
                with open(history_path, "w", encoding="utf-8") as f:
                    f.write(script_text)
                logger.info(f"Saved transcript to: {transcript_path} and history copy: {history_path}")
                
                scripts_to_synthesize[siman] = script_text
                
            except Exception as e:
                logger.error(f"Failed to generate script for Siman {siman}: {e}. Skipping to next.")
                
    except Exception as e:
        logger.error(f"Gemini client initialization failed: {e}")
        sys.exit(1)

    # 5. Text-to-Speech (TTS) Synthesis
    if args.skip_tts:
        logger.info("Skipping Text-to-Speech synthesis stage as requested by --skip-tts.")
        logger.info("Pipeline execution finished successfully (Transcripts generated).")
        return

    if not scripts_to_synthesize:
        logger.warning("No script transcripts were successfully generated. TTS stage aborted.")
        sys.exit(1)

    try:
        tts_engine = config.get_tts_engine()
        
        for siman, script_text in scripts_to_synthesize.items():
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Synthesize directly to the timestamped history audio file
            audio_history_filename = f"Yoreh_Deah_Siman_{siman}_{timestamp}.mp3"
            audio_history_path = os.path.join(config.output_dir, audio_history_filename)
            
            try:
                logger.info(f"Synthesizing script for Siman {siman} to history audio file...")
                tts_engine.synthesize(text=script_text, output_path=audio_history_path)
                logger.info(f"Generated history MP3 saved: {audio_history_path}")
                
                # Attempt to save a copy as the main audio file (Yoreh_Deah_Siman_X.mp3)
                audio_filename = f"Yoreh_Deah_Siman_{siman}.mp3"
                audio_path = os.path.join(config.output_dir, audio_filename)
                try:
                    import shutil
                    shutil.copyfile(audio_history_path, audio_path)
                    logger.info(f"Updated latest MP3 copy at: {audio_path}")
                except Exception as copy_err:
                    logger.warning(
                        f"Could not overwrite latest MP3 copy at '{audio_path}' (it may be locked by a media player): {copy_err}. "
                        f"The generated audio remains fully saved in: '{audio_history_path}'."
                    )
            except Exception as e:
                logger.error(f"TTS synthesis failed for Siman {siman}: {e}")
                
    except Exception as e:
        logger.error(f"TTS Engine initialization failed: {e}")
        sys.exit(1)

    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
