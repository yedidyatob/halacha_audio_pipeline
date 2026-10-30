"""
Cheap polish-prompt test: runs the real Stage 3 polish+nikkud call (config.yaml polishing_instruction) on
a few seifim of a siman's existing Stage 1 draft, and saves the raw polish output. Never generates a draft,
writes no cache, no dictionary nikkud, no TTS.

    python scripts/polish_sample.py 96 --seifim 1-2 [--draft-file PATH] [--relations-file PATH] [--allow-no-relations]
        [--config PATH]

Relations map: --relations-file if given, else the newest cached map of the current model whose simanim
include this siman. With neither it stops (the map changes the polished length, so a sample without one is
not representative) unless --allow-no-relations is given.
"""
import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from main import draft_base_name, lesson_base_name, model_suffix_for  # noqa: E402
from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.factory import create_generator_engine  # noqa: E402
from pipeline.gematria import gematria_to_int  # noqa: E402
from pipeline.input_parser import parse_simanim_string  # noqa: E402
from pipeline.utils import save_output_file  # noqa: E402
from pipeline.logger import get_logger  # noqa: E402

logger = get_logger("polish_sample")

# A seif heading of the Stage 1 schema (system_instruction): "## סעיף א': ..." (any heading level).
SEIF_HEADING = re.compile(r"^#+\s*סעיף\s+([א-ת'\"׳״]+)\s*[:.]?", re.MULTILINE)


class SampleError(ValueError):
    """The sample cannot be made; the message says why."""


def split_seifim(draft: str) -> tuple:
    """Draft -> (text before the first seif heading, {seif number: section text incl. its heading})."""
    matches = list(SEIF_HEADING.finditer(draft))
    if not matches:
        raise SampleError("No seif headings ('## סעיף א': ...') found in the draft; cannot split it by seif.")
    sections = {}
    for i, match in enumerate(matches):
        try:
            seif = gematria_to_int(match.group(1))
        except ValueError as e:
            raise SampleError(f"Unreadable seif heading in the draft: '{match.group(0)}'") from e
        if seif in sections:
            raise SampleError(f"Seif {seif} appears twice in the draft; cannot split it by seif.")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(draft)
        sections[seif] = draft[match.start():end].strip()
    return draft[:matches[0].start()].strip(), sections


def draft_slice(draft: str, seifim: list) -> str:
    preamble, sections = split_seifim(draft)
    missing = [s for s in seifim if s not in sections]
    if missing:
        raise SampleError(f"Seifim {missing} are not in the draft (it has {sorted(sections)}).")
    return "\n\n".join(([preamble] if preamble else []) + [sections[s] for s in seifim])


def find_cached_relations(config, siman: int, model_suffix: str):
    """The newest cached relations map of this model whose simanim include siman (history copies excluded)."""
    pattern = re.compile(rf"^{re.escape(config.section_slug)}_Relations_([0-9_]+)_{re.escape(model_suffix)}\.txt$")
    if not os.path.isdir(config.relations_dir):
        return None
    found = []
    for name in os.listdir(config.relations_dir):
        match = pattern.match(name)
        if match and siman in {int(n) for n in match.group(1).split("_")}:
            path = os.path.join(config.relations_dir, name)
            found.append((os.path.getmtime(path), path))
    return max(found)[1] if found else None


def word_count(text: str) -> int:
    return len(text.split())


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Polish a few seifim of a cached Stage 1 draft (prompt testing).")
    parser.add_argument("siman", type=str, help="The siman, e.g. 96.")
    parser.add_argument("--seifim", type=str, required=True, help="Seif, range or list, e.g. 1, 1-2, 1,3.")
    parser.add_argument("--draft-file", type=str, default=None,
                        help="Stage 1 draft to use instead of the current model's cached draft.")
    parser.add_argument("--relations-file", type=str, default=None,
                        help="Relations map to use instead of the newest cached one that includes the siman.")
    parser.add_argument("--allow-no-relations", action="store_true",
                        help="Polish without a relations map when none is given or cached for the siman.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    args = parser.parse_args(argv)
    try:
        simanim = parse_simanim_string(args.siman)
        args.seifim_list = parse_simanim_string(args.seifim)
    except ValueError as e:
        parser.error(str(e))
    if len(simanim) != 1:
        parser.error(f"expected a single siman, got '{args.siman}'")
    args.siman = simanim[0]
    return args


def _read(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise SampleError(f"{what} not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run(args) -> str:
    """Makes the sample; returns the saved output path."""
    config = PipelineConfig(args.config)
    generator = create_generator_engine(config)
    model_suffix = model_suffix_for(generator)

    draft_path = args.draft_file or os.path.join(config.drafts_dir,
                                                 draft_base_name(config, args.siman, model_suffix) + ".txt")
    draft = draft_slice(_read(draft_path, f"Stage 1 draft for Siman {args.siman}"), args.seifim_list)
    print(f"Draft: {draft_path} -> seifim {args.seifim_list} ({word_count(draft)} words)")

    relations_path = args.relations_file or find_cached_relations(config, args.siman, model_suffix)
    if not relations_path and not args.allow_no_relations:
        raise SampleError(f"No relations map covers Siman {args.siman} for model {model_suffix}; pass "
                          "--relations-file PATH, or --allow-no-relations to polish without one.")
    relations = _read(relations_path, "Relations map") if relations_path else ""
    print(f"Relations map: {relations_path or 'none (--allow-no-relations)'}")

    # the real Stage 3 call; the generator logs tokens and cost ([Gemini Cost Tracker], config gemini_pricing)
    polished = generator.polish_siman_script(draft, relations, config.polishing_instruction,
                                             temperature=config.polish_temperature)
    spec = "_".join(str(s) for s in args.seifim_list)
    base_name = f"{lesson_base_name(config, args.siman, model_suffix)}_sample_seifim_{spec}"
    save_output_file(directory=config.output_dir, base_name=base_name, extension="txt", content=polished,
                     logger=logger, label=f"Polish sample for Siman {args.siman}")
    path = os.path.join(config.output_dir, base_name + ".txt")
    print(f"Output: {path} ({word_count(polished)} words; draft slice {word_count(draft)} words)")
    return path


def main(argv=None) -> None:
    load_dotenv(dotenv_path=REPO_ROOT.parent / ".env")  # .env lives in the folder above the repo
    args = parse_arguments(argv)
    for name in ("config", "draft_file", "relations_file"):  # given relative to the caller's folder
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    os.chdir(REPO_ROOT)  # config.yaml's output paths are repo-relative
    try:
        run(args)
    except SampleError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
