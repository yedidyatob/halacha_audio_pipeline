import sys
import os
import traceback
import signal
from dotenv import load_dotenv
load_dotenv()

from pipeline.config import PipelineConfig
from pipeline.generator import GeminiScriptGenerator

def timeout_handler(signum, frame):
    raise TimeoutError("API call exceeded 60 second timeout")

try:
    config = PipelineConfig("config.yaml")
    generator = GeminiScriptGenerator(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model_name,
        temperature=config.gemini_temperature
    )
    
    # Read draft
    with open("output/drafts/Yoreh_Deah_Siman_94_gemini-3_1-pro-preview_draft.txt", "r", encoding="utf-8") as f:
        draft = f.read()
    
    # Read relations
    with open("output/relations/Yoreh_Deah_Relations_94_gemini-3_1-pro-preview.txt", "r", encoding="utf-8") as f:
        relations = f.read()
    
    # Read polishing instruction
    polishing_instruction = config.polishing_instruction
    
    print("Starting polish operation...")
    print(f"Draft length: {len(draft)} chars")
    print(f"Relations length: {len(relations)} chars")
    print(f"Polishing instruction length: {len(polishing_instruction)} chars")
    print(f"Model name: {config.gemini_model_name}")
    print(f"Temperature: {config.gemini_temperature}")
    
    # Set timeout for API call (Windows doesn't support signal.alarm)
    print("\nCalling Gemini API... (this may take a while)")
    
    try:
        result = generator.polish_siman_script(draft, relations, polishing_instruction)
        print(f"\n✓ Success! Polished script length: {len(result)} chars")
        print("\nFirst 500 chars of result:")
        print(result[:500])
    except TimeoutError as te:
        print(f"\n✗ Timeout: {te}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as api_error:
        print(f"\n✗ API call failed: {type(api_error).__name__}: {api_error}")
        traceback.print_exc()
        sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Error occurred: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
