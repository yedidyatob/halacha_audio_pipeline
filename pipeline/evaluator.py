import re
import requests
from typing import Dict, List, Any, Set, Tuple, Optional
from pipeline.config import PipelineConfig
from pipeline.extractor import SefariaExtractor
from pipeline.gematria import int_to_gematria
from pipeline.logger import get_logger

# Logger initialization

logger = get_logger(__name__)

def get_clean_gematria_without_quotes(num: int) -> str:
    """Helper to get gematria letters without quotes (e.g., צד instead of צ"ד)"""
    g = int_to_gematria(num)
    return g.replace('"', '').replace("'", "")

def get_seif_from_ref(ref_str: str, siman: int, sefaria_name: str = "Yoreh De'ah") -> Optional[int]:
    """Parses the Shulchan Arukh Se'if number from a Sefaria reference string"""
    escaped_name = re.escape(sefaria_name)
    match = re.search(rf"Shulchan Arukh, {escaped_name} {siman}:(\d+)", ref_str)
    if match:
        return int(match.group(1))
    return None

def parse_draft_into_seifim(draft_text: str, num_seifim: int) -> Dict[int, str]:
    """Splits the draft text into individual blocks by Se'if headers"""
    seif_blocks = {}
    
    # Locate all markdown headers for Se'ifim
    # Matches "## סעיף א", "## סעיף א (...", "## סעיף 1", etc.
    header_pattern = r"##\s+(סעיף|בסעיף)\s+[\(\'\"]*([א-ת]+|\d+)[\)\'\"]*"
    matches = list(re.finditer(header_pattern, draft_text))
    
    for idx, match in enumerate(matches):
        seif_indicator = match.group(2)
        
        # Match indicator to Se'if number
        seif_num = None
        for i in range(1, num_seifim + 1):
            g_with_quotes = int_to_gematria(i)
            g_without_quotes = get_clean_gematria_without_quotes(i)
            if seif_indicator in [g_with_quotes, g_without_quotes, str(i)]:
                seif_num = i
                break
                
        if seif_num is not None:
            start_pos = match.end()
            end_pos = matches[idx+1].start() if idx + 1 < len(matches) else len(draft_text)
            seif_blocks[seif_num] = draft_text[start_pos:end_pos]
            
    return seif_blocks

def evaluate_draft(siman: int, draft_text: str, config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Evaluates a generated Stage 1 draft locally using zero-token heuristic checks.
    Compares the draft against Sefaria's active commentators on a per-Se'if and global level.
    
    Returns a dictionary:
    {
        "success": bool,
        "total_localized_flags": int,
        "total_general_flags": int,
        "missing_seifim": List[int],
        "report": str
    }
    """
    report_lines = []
    report_lines.append(f"============================================================")
    report_lines.append(f"      LOCALIZED HEURISTIC EVALUATION FOR SIMAN {siman}")
    report_lines.append(f"============================================================\n")

    # 1. Fetch Raw Sources and Links from Sefaria
    config = PipelineConfig(config_path)
    if not config.ssl_verify:
        requests.urllib3.disable_warnings(requests.urllib3.exceptions.InsecureRequestWarning)

    extractor = SefariaExtractor(
        section_name=config.halachic_section,
        base_url=config.sefaria_base_url,
        timeout=config.sefaria_timeout,
        retries=config.sefaria_retries,
        ssl_verify=config.ssl_verify
    )
    
    report_lines.append("Fetching raw Hebrew sources from Sefaria API...")
    sources = extractor.fetch_siman_sources(siman)
    
    # Fetch links to map commentators to Se'ifim
    sefaria_name_for_url = config.section_metadata['sefaria_name'].replace(" ", "_")
    links_url = f"{config.sefaria_base_url}/links/Shulchan_Arukh,_{sefaria_name_for_url}.{siman}"
    links = []
    try:
        response = requests.get(links_url, timeout=15, verify=config.ssl_verify)
        if response.status_code == 200:
            links = response.json()
            report_lines.append(f"Fetched {len(links)} cross-reference links from Sefaria.")
        else:
            report_lines.append(f"Warning: Failed to fetch links (Status {response.status_code}).")
    except Exception as e:
        report_lines.append(f"Warning: Error fetching links: {e}")
    report_lines.append("Fetch complete.\n")

    shulchan_arukh_lines = sources.get("Shulchan Arukh", [])
    num_seifim = len(shulchan_arukh_lines)
    
    if num_seifim == 0:
        return {
            "success": False,
            "total_localized_flags": 0,
            "total_general_flags": 0,
            "missing_seifim": [],
            "report": "Error: Shulchan Arukh text was not found for this Siman in Sefaria."
        }

    # 2. Map Sefaria Commentators to each Se'if
    commentators_by_seif = {i: set() for i in range(1, num_seifim + 1)}
    
    for link in links:
        ref = link.get("ref", "")
        anchorRef = link.get("anchorRef", "")
        
        sa_ref = None
        comm_ref = None
        
        sa_prefix = f"Shulchan Arukh, {config.section_metadata['sefaria_name']}"
        if ref.startswith(sa_prefix):
            sa_ref = ref
            comm_ref = anchorRef
        elif anchorRef.startswith(sa_prefix):
            sa_ref = anchorRef
            comm_ref = ref
            
        if not sa_ref or not comm_ref:
            continue
            
        seif = get_seif_from_ref(sa_ref, siman, config.section_metadata['sefaria_name'])
        if not seif or seif < 1 or seif > num_seifim:
            continue
            
        for comm_id, comm_info in config.section_metadata["commentators"].items():
            if comm_info["sefaria_name"] in comm_ref or comm_ref.startswith(comm_info["sefaria_name"]):
                commentators_by_seif[seif].add(comm_id)

    # Add Mechaber and Rema presence (Tur and Beit Yosef are checked globally at the Siman level)
    for i in range(1, num_seifim + 1):
        commentators_by_seif[i].add("Mechaber")
        
        # Check if Rema has a gloss in this Se'if
        seif_text = shulchan_arukh_lines[i-1]
        if "הגה" in seif_text:
            commentators_by_seif[i].add("Rema")

    # 3. Parse the Draft into Se'ifim blocks
    draft_seifim_blocks = parse_draft_into_seifim(draft_text, num_seifim)

    # 4. Commentator patterns for regex search
    commentators_patterns = {
        "Tur": r"(טור|הטור|בעל הטורים)",
        "Beit Yosef": r"(בית יוסף|ב[״\"׳']י|הבית יוסף)",
        "Mechaber": r"(מחבר|שולחן ערוך|השולחן ערוך|ש[ו״\"׳']{1,2}ע|הש[ו״\"׳']{1,2}ע|מרן|המחבר)",
        "Rema": r"(רמ[״\"׳']א|הרמ[״\"׳']א|רמ[״\"׳']א בהגה|הגה|הגהת הרמ[״\"׳']א|הגהות הרמ[״\"׳']א)",
    }
    for comm_id, comm_info in config.section_metadata["commentators"].items():
        commentators_patterns[comm_id] = comm_info["pattern"]

    # 5. Run Localized Checks
    report_lines.append("--- SE'IF-BY-SE'IF LOCALIZED COMMENTATOR COVERAGE CHECK ---")
    
    localized_flags = 0
    missing_seifim_blocks = []

    for i in range(1, num_seifim + 1):
        g_letter = get_clean_gematria_without_quotes(i)
        report_lines.append(f"\n[סעיף {g_letter}] (Se'if {i})")
        
        # Check if draft has a block for this Se'if
        block_text = draft_seifim_blocks.get(i)
        if not block_text:
            report_lines.append("  ⚠️ Draft has no section block for this Se'if!")
            missing_seifim_blocks.append(i)
            localized_flags += 1
            continue
            
        # Determine required commentators for this Se'if
        required_commentators = commentators_by_seif[i]
        required_list = sorted(list(required_commentators))
        report_lines.append(f"  Commentators active in Sefaria source: {', '.join(required_list)}")
        
        for comm in required_list:
            pattern = commentators_patterns[comm]
            matches = re.findall(pattern, block_text, re.IGNORECASE)
            count = len(matches)
            
            if count > 0:
                status = "[PASS]"
            else:
                status = "[WARN] (Missing!)"
                localized_flags += 1
                
            report_lines.append(f"    - {comm:<12}: {count:<3} mentions {status}")

    # 6. Run General/General Checks for Tur and Beit Yosef
    report_lines.append("\n--- GENERAL DRAFT COVERAGE CHECK ---")
    general_flags = 0
    
    for comm in ["Tur", "Beit Yosef"]:
        pattern = commentators_patterns[comm]
        matches = re.findall(pattern, draft_text, re.IGNORECASE)
        count = len(matches)
        if count > 0:
            status = "[PASS]"
        else:
            status = "[WARN] (Missing from entire draft!)"
            general_flags += 1
            
        report_lines.append(f"  - {comm:<12}: {count:<3} mentions {status}")

    total_flags = localized_flags + general_flags
    success = (total_flags == 0)

    report_lines.append("\n============================================================")
    report_lines.append("--- FINAL SUMMARY REPORT ---")
    report_lines.append(f"Total Localized Omission Flags: {localized_flags}")
    report_lines.append(f"Total General Omission Flags:   {general_flags}")
    if missing_seifim_blocks:
        report_lines.append(f"Missing Se'ifim blocks:        {missing_seifim_blocks}")
        
    if success:
        report_lines.append("Status: EXCELLENT (100% Localized Heuristic Coverage Pass)")
    else:
        report_lines.append(f"Status: WARNING ({total_flags} issues identified)")
        report_lines.append("Action: Please review the flagged items where active commentators were not mentioned in their corresponding Se'if draft block or globally.")
    report_lines.append("============================================================")

    return {
        "success": success,
        "total_localized_flags": localized_flags,
        "total_general_flags": general_flags,
        "missing_seifim": missing_seifim_blocks,
        "report": "\n".join(report_lines)
    }
