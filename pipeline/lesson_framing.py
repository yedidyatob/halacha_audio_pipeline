"""Fixed lesson intro/outro framing for TTS lessons.

The intro and outro are plain templates in ``config.yaml`` (``lesson_framing.intro`` /
``lesson_framing.outro``), rendered per siman and glued around the polished lesson body in
code, so the LLM never writes the greeting/closing itself.

Placeholders (all optional in a template):
    {hebrew_section}  Hebrew name of the section, e.g. יורה דעה
    {spoken_siman}    siman as spoken letter names, e.g. 94 -> "צדי דלת" (see gematria.LETTER_NAMES)
    {gematria_siman}  siman as written gematria, e.g. 94 -> צ"ד
    {siman}           siman as a number, e.g. 94
"""

from __future__ import annotations

from pipeline.gematria import int_to_gematria, int_to_spoken_gematria

PLACEHOLDERS = ("hebrew_section", "spoken_siman", "gematria_siman", "siman")


def validate_lesson_template(name: str, template: str) -> None:
    """
    Raise ValueError with a clear message if ``template`` is not a valid frame template
    (unknown placeholder, stray brace). Used at config load so errors surface at startup.
    """
    try:
        _format(template, siman=1, hebrew_section="")
    except KeyError as e:
        raise ValueError(
            f"Unknown placeholder {{{e.args[0]}}} in 'lesson_framing.{name}' in config.yaml. "
            f"Allowed placeholders: {', '.join('{' + p + '}' for p in PLACEHOLDERS)}."
        ) from None
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid template in 'lesson_framing.{name}' in config.yaml: {e}") from None


def _format(template: str, siman: int, hebrew_section: str) -> str:
    return template.format(
        hebrew_section=hebrew_section,
        gematria_siman=int_to_gematria(siman),
        spoken_siman=int_to_spoken_gematria(siman),
        siman=siman,
    )


def render_lesson_frame(template: str, siman: int, hebrew_section: str) -> str:
    """Fill the placeholders of a lesson intro/outro template ('' for an empty template)."""
    if not template or not template.strip():
        return ""
    if isinstance(siman, bool) or not isinstance(siman, int):
        raise TypeError(
            f"Lesson framing needs a single siman number, got {siman!r}. "
            "Simanim ranges are rendered one lesson (one siman) at a time."
        )
    return _format(template, siman, hebrew_section).strip()


def wrap_lesson_with_frames(
    body: str,
    siman: int,
    hebrew_section: str,
    intro_template: str,
    outro_template: str,
) -> str:
    """
    Prepend/append the fixed intro and outro around the polished lesson body, separated by
    blank lines (paragraph breaks for the TTS chunker). Empty templates are skipped.
    """
    parts = []
    intro = render_lesson_frame(intro_template, siman, hebrew_section)
    if intro:
        parts.append(intro)
    body = (body or "").strip()
    if body:
        parts.append(body)
    outro = render_lesson_frame(outro_template, siman, hebrew_section)
    if outro:
        parts.append(outro)
    return "\n\n".join(parts)
