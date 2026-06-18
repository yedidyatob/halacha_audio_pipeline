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
