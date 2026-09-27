# Coding Standards: Halacha Audio Pipeline

## Python Conventions

- Use `snake_case` for functions and variables
- Use `PascalCase` for class names
- Use module-level `get_logger(__name__)` for all logging
- Include type hints for all function signatures

## Error Handling

- All API calls must be wrapped in try/except blocks
- Log errors with `logger.error()` before re-raising or exiting
- Exit with `sys.exit(1)` on recoverable failures
- Never let exceptions propagate past `main()` without handling

## File Organization

- All pipeline modules live in `pipeline/` directory
- Keep utility functions in `pipeline/utils.py` only if used by ≥2 modules
- Add new modules for distinct responsibilities (extractor, generator, tts)

## Code Quality

- Follow DRY: consolidate duplicate logic into functions like `save_output_file()`
- Use constants for magic strings (e.g. `Yoreh_Deah`, `siman_94`)
- Comment complex logic blocks but avoid obvious comments
- Prefer `logger.info()` over `print()` statements

## Hebrew Text Requirements

- All prompts and system instructions are in Hebrew
- TTS output scripts must be Hebrew-only (no code-switching)
- Gematria conversions use `pipeline/gematria.py` (int_to_gematria)
- Hebrew abbreviations must be expanded for TTS (e.g. "כ"י" → "כביון יסוד")
