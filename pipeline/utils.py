import os
import shutil
from datetime import datetime
from typing import Union, Callable, Optional

def save_output_file(
    directory: str,
    base_name: str,
    extension: str,
    content: Optional[Union[str, bytes]],
    logger,
    write_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Saves an output file in two versions:
    1. A timestamped history version (which is never locked and acts as backup).
    2. A 'latest' copy (which might be locked by an external editor/player).
    
    If write_callback is provided, it is invoked with the history path as its sole argument
    to write the file (e.g. for TTS synthesis). Otherwise, 'content' is written.
    
    Returns the path to the history file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(directory, exist_ok=True)
    
    history_filename = f"{base_name}_{timestamp}.{extension}"
    latest_filename = f"{base_name}.{extension}"
    
    history_path = os.path.join(directory, history_filename)
    latest_path = os.path.join(directory, latest_filename)
    
    # 1. Save history first
    try:
        if write_callback:
            write_callback(history_path)
        else:
            mode = "wb" if isinstance(content, bytes) else "w"
            encoding = None if isinstance(content, bytes) else "utf-8"
            with open(history_path, mode, encoding=encoding) as f:
                f.write(content)
        logger.info(f"Saved history copy: {history_path}")
    except Exception as e:
        logger.error(f"Failed to save history copy to '{history_path}': {e}")
        raise
        
    # 2. Attempt to update latest copy
    try:
        shutil.copyfile(history_path, latest_path)
        logger.info(f"Updated latest copy at: {latest_path}")
    except Exception as err:
        logger.warning(
            f"Could not overwrite latest copy at '{latest_path}' (it may be locked by another application): {err}. "
            f"The generated content remains fully saved in: '{history_path}'."
        )
        
    return history_path


def log_gemini_usage(logger, response, model_name: str) -> None:
    """
    Extracts usage_metadata from the Gemini response, calculates the estimated
    cost on the fly based on current pricing, and logs it.
    """
    if not hasattr(response, "usage_metadata") or not response.usage_metadata:
        return
        
    prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
    output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
    total_tokens = getattr(response.usage_metadata, "total_token_count", 0)
    
    # Handle mock objects in unit tests gracefully
    if not isinstance(prompt_tokens, (int, float)) or not isinstance(output_tokens, (int, float)):
        return
    
    # Determine pricing based on model class
    model_lower = model_name.lower()
    if "tts" in model_lower:
        # Gemini Flash TTS pricing: $0.075/1M input, $20.00/1M audio output
        input_rate = 0.075 / 1_000_000
        output_rate = 20.00 / 1_000_000
    elif "pro" in model_lower:
        # Gemini Pro pricing: $1.25/1M input, $5.00/1M output
        input_rate = 1.25 / 1_000_000
        output_rate = 5.00 / 1_000_000
    else:
        # Gemini Flash pricing: $0.075/1M input, $0.30/1M output
        input_rate = 0.075 / 1_000_000
        output_rate = 0.30 / 1_000_000
        
    estimated_cost = (prompt_tokens * input_rate) + (output_tokens * output_rate)
    
    logger.info(
        f"[Gemini Cost Tracker] Model: {model_name} | "
        f"Tokens Used: Input={prompt_tokens}, Output={output_tokens}, Total={total_tokens} | "
        f"Estimated Cost: ${estimated_cost:.6f} USD"
    )
