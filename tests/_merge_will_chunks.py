"""
Merge the will_test audio chunks in timestamp order using the fixed _merge_audio_chunks method.
"""
import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.tts import BaseTTS


class _ConcreteTTS(BaseTTS):
    """Minimal concrete subclass so we can call _merge_audio_chunks directly."""
    _chunk_size = 9999

    def _setup(self): pass
    def _synthesize_chunk(self, text: str) -> bytes: return b""


def main():
    # Get all mp3 files in output/will_test
    chunk_dir = "output/will_test"
    chunks = sorted(glob.glob(os.path.join(chunk_dir, "*.mp3")))
    
    if not chunks:
        print(f"No MP3 files found in {chunk_dir}")
        sys.exit(1)
    
    print(f"Found {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. {os.path.basename(chunk)}")
    
    output_path = "output/will_test_merged.mp3"
    
    print(f"\nMerging into: {output_path}")
    tts = _ConcreteTTS()
    tts._merge_audio_chunks(chunks, output_path)
    
    print(f"✓ Done! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
