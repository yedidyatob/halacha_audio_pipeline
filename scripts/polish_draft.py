"""
Experiment: run ONLY the Stage 3 polish call on an existing Stage 1 draft + Stage 2 relations
map, with the generator from config.yaml (generator.engine / model), exactly as the pipeline
builds the request (BaseScriptGenerator.polish). Prints token usage and an estimated cost
(prices from experiment_pricing in config.yaml).

Usage (from the repo root):
  python scripts/polish_draft.py output/drafts/<draft>.txt output/relations/<relations>.txt [--output out.txt]

Default output: <draft stem>_polished.txt next to the draft. No nikkud and no pronunciation
tags are applied; run scripts/tag_pronunciation.py on the result for the two-stage comparison.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Polish a Stage 1 draft (Stage 3 only) and print token usage.")
    parser.add_argument("draft", help="Stage 1 draft file (output/drafts/...)")
    parser.add_argument("relations", help="Stage 2 relations file (output/relations/...)")
    parser.add_argument("--output", default=None, help="Output path (default: <draft stem>_polished.txt)")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: <repo>/config.yaml)")
    args = parser.parse_args(argv)
    args.draft = os.path.abspath(args.draft)
    args.relations = os.path.abspath(args.relations)
    args.output = os.path.abspath(args.output) if args.output else f"{os.path.splitext(args.draft)[0]}_polished.txt"
    args.config = os.path.abspath(args.config) if args.config else str(REPO_ROOT / "config.yaml")
    return args


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=REPO_ROOT.parent / ".env")  # .env lives in the folder above the repo

    from pipeline.config import PipelineConfig
    from pipeline.factory import create_generator_engine
    from pipeline.usage import format_usage_report
    from scripts.polish_and_tag_combined import read_inputs

    draft, relations = read_inputs(args.draft, args.relations)
    config = PipelineConfig(args.config)
    generator = create_generator_engine(config)
    pricing = config.experiment_pricing(generator.model_name)  # fail before paying for a call

    print(f"Polishing {args.draft} via {config.generator_engine} / {generator.model_name} (thinking level: model default)...")
    result = generator.polish(draft, relations, config.polishing_instruction)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result.text)
    print(f"Polished script saved to: {args.output}")
    print(format_usage_report("Polish", generator.model_name, result.usage, pricing))


if __name__ == "__main__":
    main()
