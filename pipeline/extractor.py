import html
import re
import requests
import time
import urllib3
from typing import List, Dict, Any, Optional
from pipeline.logger import get_logger

# Disable SSL verification warnings for Sefaria API requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = get_logger(__name__)

class SefariaExtractor:
    """
    Fetches and cleans classical Jewish texts from the Sefaria API.
    Handles nested list structures and HTML sanitization.
    """
    def __init__(self, base_url: str = "https://www.sefaria.org/api", timeout: int = 15, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        
        # Sefaria canonical text references for Yoreh De'ah
        self.works = {
            "Tur": "Tur, Yoreh De'ah",
            "Beit Yosef": "Beit Yosef, Yoreh De'ah",
            "Shulchan Arukh": "Shulchan Arukh, Yoreh De'ah",
            "Shach": "Siftei Kohen on Shulchan Arukh, Yoreh De'ah",
            "Taz": "Turei Zahav on Shulchan Arukh, Yoreh De'ah"
        }

    def clean_html_text(self, text: str) -> str:
        """
        Removes HTML tags, unescapes HTML entities, and strips extra spaces.
        """
        if not text:
            return ""
        # Remove HTML tags (e.g., <b>, <i>, data commentators, etc.)
        clean = re.sub(r"<[^>]+>", "", text)
        # Unescape HTML entities
        clean = html.unescape(clean)
        # Normalize whitespace
        return " ".join(clean.split()).strip()

    def flatten_elements(self, data: Any) -> List[str]:
        """
        Recursively flattens nested lists of strings into a flat list of cleaned strings.
        """
        if isinstance(data, str):
            cleaned = self.clean_html_text(data)
            return [cleaned] if cleaned else []
        elif isinstance(data, list):
            flat = []
            for item in data:
                flat.extend(self.flatten_elements(item))
            return flat
        return []

    def fetch_text(self, ref: str) -> List[str]:
        """
        Fetches text from the Sefaria API using a retry loop.
        First attempts the v3 endpoint, falling back to the v1 endpoint.
        """
        # Replace spaces with underscores for URL encoding
        encoded_ref = ref.replace(" ", "_")
        
        # 1. Try v3 API first
        v3_url = f"{self.base_url}/v3/texts/{encoded_ref}?version=hebrew"
        for attempt in range(self.retries):
            try:
                logger.info(f"Fetching (v3) ref '{ref}' (Attempt {attempt + 1}/{self.retries})...")
                # Using verify=False to bypass certificate errors in environments with SSL inspection
                response = requests.get(v3_url, timeout=self.timeout, verify=False)
                if response.status_code == 200:
                    payload = response.json()
                    versions = payload.get("versions", [])
                    
                    # Find a Hebrew version
                    he_version = next((v for v in versions if v.get("language") == "he"), None)
                    if he_version and "text" in he_version:
                        flat_text = self.flatten_elements(he_version["text"])
                        if flat_text:
                            return flat_text
                            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Sefaria API v3 error for '{ref}' on attempt {attempt + 1}: {e}")
                time.sleep(1)

        # 2. Fall back to v1 API
        v1_url = f"{self.base_url}/texts/{encoded_ref}?context=0"
        for attempt in range(self.retries):
            try:
                logger.info(f"Falling back (v1) for ref '{ref}' (Attempt {attempt + 1}/{self.retries})...")
                # Using verify=False to bypass certificate errors in environments with SSL inspection
                response = requests.get(v1_url, timeout=self.timeout, verify=False)
                if response.status_code == 200:
                    payload = response.json()
                    hebrew_data = payload.get("he", [])
                    flat_text = self.flatten_elements(hebrew_data)
                    if flat_text:
                        return flat_text
            except requests.exceptions.RequestException as e:
                logger.warning(f"Sefaria API v1 error for '{ref}' on attempt {attempt + 1}: {e}")
                time.sleep(1)

        logger.error(f"Failed to fetch any Hebrew text for reference: {ref}")
        return []

    def fetch_siman_sources(self, siman: int) -> Dict[str, List[str]]:
        """
        Fetches all five works for a specific Siman.
        """
        siman_data = {}
        for work_name, base_ref in self.works.items():
            ref = f"{base_ref}.{siman}"
            text_lines = self.fetch_text(ref)
            if text_lines:
                siman_data[work_name] = text_lines
            else:
                logger.warning(f"Work '{work_name}' was empty or not found for Siman {siman}")
                siman_data[work_name] = []
        return siman_data

    def compile_simanim_context(self, simanim: List[int]) -> str:
        """
        Extracts and compiles a range of Simanim into a single structured master text context.
        """
        compiled_sections = []
        
        for siman in simanim:
            logger.info(f"Starting Sefaria extraction for Siman {siman}...")
            siman_header = (
                f"\n\n=========================================\n"
                f"=== סימן {siman} ===\n"
                f"=========================================\n\n"
            )
            compiled_sections.append(siman_header)
            
            siman_sources = self.fetch_siman_sources(siman)
            
            # Format and append each work to the context file
            for work_name in ["Tur", "Beit Yosef", "Shulchan Arukh", "Shach", "Taz"]:
                lines = siman_sources.get(work_name, [])
                if lines:
                    section_title = f"--- {work_name} (סימן {siman}) ---\n\n"
                    compiled_sections.append(section_title)
                    for idx, line in enumerate(lines, 1):
                        compiled_sections.append(f"[{idx}] {line}\n\n")
            
            logger.info(f"Completed Sefaria extraction for Siman {siman}.")

        return "".join(compiled_sections)
