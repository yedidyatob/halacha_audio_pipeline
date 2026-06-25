import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured Logger instance.
    Logs to stdout with a clean, readable format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Reconfigure stdout/stderr encoding on Windows to prevent UnicodeEncodeError when printing emojis/Hebrew
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="backslashreplace")
                except Exception:
                    pass
        
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)
    return logger

