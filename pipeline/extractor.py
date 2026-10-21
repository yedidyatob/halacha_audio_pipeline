import html
import re
import requests
import time
import urllib3
from typing import List, Dict, Any, Optional
from pipeline.logger import get_logger
logger = get_logger(__name__)

# Sefaria prints the Rama's glosses on the Shulchan Arukh in <small> (the Rama's smaller font):
# both the glosses that open with "הגה" and the short inline ones inside the Mechaber's text.
# Each top-level <small> block becomes "[רמ"א] ... [/רמ"א]" before tags are stripped, so the
# attribution survives in plain text. An explicit closing marker is used because a gloss can
# itself contain square brackets (e.g. "[נ"ט]"); it is never numeric, so it cannot be confused
# with the "[n]" se'if numbering of the compiled context.
RAMA_OPEN = '[רמ"א]'
RAMA_CLOSE = '[/רמ"א]'
_SMALL_TAG_RE = re.compile(r"<(/?)small\b[^>]*>", re.IGNORECASE)
_LEADING_HAGAH_RE = re.compile(r"^הגה\s*:?\s*")

# The work whose <small> blocks are Rama glosses.
SHULCHAN_ARUKH = "Shulchan Arukh"


class SefariaExtractor:
    """
    Fetches and cleans classical Jewish texts from the Sefaria API.
    Handles nested list structures and HTML sanitization.
    """
    def __init__(self, section_name: str = "Yoreh De'ah", base_url: str = "https://www.sefaria.org/api", timeout: int = 15, retries: int = 3, ssl_verify: bool = True):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.ssl_verify = ssl_verify
        self.section_name = section_name
        
        if not self.ssl_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        from pipeline.domain import SECTIONS_METADATA
        if section_name not in SECTIONS_METADATA:
            raise ValueError(f"Unsupported section name: '{section_name}'")
            
        metadata = SECTIONS_METADATA[section_name]
        sefaria_name = metadata["sefaria_name"]
        
        # Sefaria canonical text references resolved dynamically
        self.works = {
            "Tur": f"Tur, {sefaria_name}",
            "Beit Yosef": f"Beit Yosef, {sefaria_name}",
            SHULCHAN_ARUKH: f"Shulchan Arukh, {sefaria_name}"
        }
        for comm_name, comm_info in metadata["commentators"].items():
            self.works[comm_name] = comm_info["sefaria_name"]

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

    def label_rama_glosses(self, text: str) -> str:
        """
        Replaces each top-level <small>...</small> (a Rama gloss) with
        '[רמ"א] gloss [/רמ"א]', the gloss cleaned of tags and of a leading "הגה".
        Nested <small> tags belong to the enclosing gloss; an unclosed one runs to the end.
        """
        parts, depth, pos, start = [], 0, 0, 0
        for tag in _SMALL_TAG_RE.finditer(text):
            if not tag.group(1):
                if depth == 0:
                    parts.append(text[pos:tag.start()])
                    start = tag.end()
                depth += 1
            elif depth:
                depth -= 1
                if depth == 0:
                    parts.append(self._rama_label(text[start:tag.start()]))
                    pos = tag.end()
        if depth:
            parts.append(self._rama_label(text[start:]))
            pos = len(text)
        parts.append(text[pos:])
        return "".join(parts)

    def _rama_label(self, gloss_html: str) -> str:
        gloss = _LEADING_HAGAH_RE.sub("", self.clean_html_text(gloss_html))
        return f" {RAMA_OPEN} {gloss} {RAMA_CLOSE} " if gloss else " "

    def flatten_elements(self, data: Any, label_rama: bool = False) -> List[str]:
        """
        Recursively flattens nested lists of strings into a flat list of cleaned strings.
        With label_rama (Shulchan Arukh text), Rama glosses are labeled before cleaning.
        """
        if isinstance(data, str):
            cleaned = self.clean_html_text(self.label_rama_glosses(data) if label_rama else data)
            return [cleaned] if cleaned else []
        elif isinstance(data, list):
            flat = []
            for item in data:
                flat.extend(self.flatten_elements(item, label_rama))
            return flat
        return []

    def fetch_work(self, work_name: str, siman: int) -> List[str]:
        """Cleaned lines of one work for a siman; Shulchan Arukh lines carry Rama labels."""
        return self.fetch_text(f"{self.works[work_name]}.{siman}", label_rama=work_name == SHULCHAN_ARUKH)

    def fetch_text(self, ref: str, label_rama: bool = False) -> List[str]:
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
                response = requests.get(v3_url, timeout=self.timeout, verify=self.ssl_verify)
                if response.status_code == 200:
                    payload = response.json()
                    versions = payload.get("versions", [])
                    
                    # Find a Hebrew version
                    he_version = next((v for v in versions if v.get("language") == "he"), None)
                    if he_version and "text" in he_version:
                        flat_text = self.flatten_elements(he_version["text"], label_rama)
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
                response = requests.get(v1_url, timeout=self.timeout, verify=self.ssl_verify)
                if response.status_code == 200:
                    payload = response.json()
                    hebrew_data = payload.get("he", [])
                    flat_text = self.flatten_elements(hebrew_data, label_rama)
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
        for work_name in self.works:
            text_lines = self.fetch_work(work_name, siman)
            if text_lines:
                siman_data[work_name] = text_lines
            else:
                logger.warning(f"Work '{work_name}' was empty or not found for Siman {siman}")
                siman_data[work_name] = []
        return siman_data

    def compile_simanim_context(self, simanim: List[int], target_simanim: Optional[List[int]] = None) -> str:
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
            
            # Determine works to fetch/include for this Siman
            is_target = target_simanim is None or siman in target_simanim
            if is_target:
                works_to_include = ["Tur", "Beit Yosef", "Shulchan Arukh", "Shach", "Taz"]
            else:
                works_to_include = ["Tur", "Shulchan Arukh"]
                logger.info(f"Siman {siman} is context-only. Fetching only Tur and Shulchan Arukh to optimize context length.")
                
            siman_sources = {}
            for work_name in works_to_include:
                text_lines = self.fetch_work(work_name, siman)
                if text_lines:
                    siman_sources[work_name] = text_lines
                else:
                    logger.warning(f"Work '{work_name}' was empty or not found for Siman {siman}")
                    siman_sources[work_name] = []
            
            # Format and append each work to the context file
            for work_name in works_to_include:
                lines = siman_sources.get(work_name, [])
                if lines:
                    section_title = f"--- {work_name} (סימן {siman}) ---\n\n"
                    compiled_sections.append(section_title)
                    for idx, line in enumerate(lines, 1):
                        compiled_sections.append(f"[{idx}] {line}\n\n")
            
            logger.info(f"Completed Sefaria extraction for Siman {siman}.")

        return "".join(compiled_sections)
