"""Fixed lesson intro/outro framing for TTS lessons."""

from __future__ import annotations

from pipeline.gematria import int_to_gematria, int_to_spoken_gematria


def render_lesson_frame(template: str, siman: int, hebrew_section: str) -> str:
    """Fill placeholders in a lesson intro/outro template."""
    if not template or not template.strip():
        return ""
    return template.format(
        hebrew_section=hebrew_section,
        gematria_siman=int_to_gematria(siman),
        spoken_siman=int_to_spoken_gematria(siman),
        siman=siman,
    ).strip()


def wrap_lesson_with_frames(
    body: str,
    siman: int,
    hebrew_section: str,
    intro_template: str,
    outro_template: str,
) -> str:
    """
    Prepend/append fixed intro and outro around the polished lesson body.
    Empty templates are skipped.
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
