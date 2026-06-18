import os
import sys
from pipeline.config import PipelineConfig
from pipeline.extractor import SefariaExtractor
from pipeline.input_parser import parse_simanim_string

def main():
    siman = 94
    config_path = "config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
        
    config = PipelineConfig(config_path)
    extractor = SefariaExtractor(
        base_url=config.sefaria_base_url,
        timeout=config.sefaria_timeout,
        retries=config.sefaria_retries
    )
    
    print(f"Fetching text from Sefaria for Siman {siman}...")
    sources = extractor.fetch_siman_sources(siman)
    
    total_chars = 0
    total_words = 0
    
    print("\n--- Sefaria Extraction Stats ---")
    for work, lines in sources.items():
        work_chars = sum(len(line) for line in lines)
        work_words = sum(len(line.split()) for line in lines)
        total_chars += work_chars
        total_words += work_words
        print(f"* {work}: {len(lines)} segments, {work_chars} chars, {work_words} words")
        
    estimated_tokens = int(total_words * 1.8)
    print(f"\nTotal Context Size:")
    print(f"- Characters: {total_chars}")
    print(f"- Words: {total_words}")
    print(f"- Estimated Gemini Input Tokens: ~{estimated_tokens}")
    
    # Save the consolidated text
    compiled_text = extractor.compile_simanim_context([siman])
    os.makedirs("debug", exist_ok=True)
    with open("debug/siman_94_context.txt", "w", encoding="utf-8") as f:
        f.write(compiled_text)
    print("\nConsolidated context saved to 'debug/siman_94_context.txt'")

if __name__ == "__main__":
    main()
