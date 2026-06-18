import os
import sys
from pipeline.evaluator import evaluate_draft
from pipeline.logger import get_logger

logger = get_logger(__name__)

def evaluate_siman(siman: int, draft_path: str):
    if not os.path.exists(draft_path):
        logger.error(f"Error: Draft file not found at {draft_path}")
        sys.exit(1)
        
    with open(draft_path, "r", encoding="utf-8") as f:
        draft_text = f.read()
        
    result = evaluate_draft(siman, draft_text, config_path="config.yaml")
    logger.info(result["report"])
    
    if not result["success"]:
        sys.exit(1)

if __name__ == "__main__":
    target_siman = 94
    draft_file = r"output\drafts\Yoreh_Deah_Siman_94_gpt-5_2_draft.txt"
    
    evaluate_siman(target_siman, draft_file)
