# Halacha Audio Lesson Generation Pipeline

An end-to-end Python pipeline that downloads classical Jewish Halachic texts from Sefaria, generates TTS-optimized Hebrew scripts using Gemini or OpenAI, and synthesizes them to high-quality audio using ElevenLabs, Google Cloud TTS, or OpenAI TTS.

Designed for advanced students preparing for Rabbinical Ordination exams, producing authentic-sounding monologue lessons with spoken transitions, definitions, and cross-references.

---

## Project Structure

```
halacha_audio_pipeline/
│
├── main.py               # CLI entry point
├── config.yaml           # Configuration file (see config.example.yaml for reference)
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes credentials, caches, outputs
│
├── pipeline/             # Core pipeline modules
│   ├── config.py         # Configuration loader & validator
│   ├── extractor.py      # Sefaria API client
│   ├── generator.py      # Gemini & OpenAI API wrappers
│   ├── tts.py            # TTS interface (ElevenLabs, OpenAI, Google, Gemini)
│   ├── factory.py        # Engine factory functions
│   ├── domain.py         # Section metadata and abbreviations
│   ├── utils.py          # Utility functions
│   └── ...               # Other modules
│
└── tests/                # Pytest suite
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file with your API keys:

```bash
# .env
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/google-credentials.json
```

Create `config.yaml` based on the structure in `config.example.yaml` (copy that file and fill in your details).

### 3. Run Your First Lesson

```bash
# Generate audio for Simanim 94-97
python main.py 94,95-97
```

---

## Pipeline Stages

The pipeline runs in three stages:

| Stage | Description |
|-------|-------------|
| **Stage 1** | Extract Halachic content from Sefaria API |
| **Stage 2** | Analyze cross-simanim relations for context |
| **Stage 3** | Generate polished TTS-ready script & synthesize audio |
| **Stage 3.4** | Pronunciation tagging (optional, `pronunciation.enabled`): an LLM wraps likely-mispronounced words in `<phoneme alphabet="ipa" ph="...">word</phoneme>` tags |
| **Stage 3.5** | Nikkud for rabbinic abbreviations (never touches text inside phoneme tags) |

### Pronunciation tags (Stage 3.4)

Configured in `config.yaml` (`pronunciation:` section + `pronunciation_instruction`; no code defaults).
`pronunciation.thinking_level` (`low`/`medium`/`high`) sets the Gemini thinking level for the tagging call
only; it is required when `pronunciation.engine` is `gemini` and ignored for `openai`.
`pronunciation.always_tag` / `pronunciation.never_tag` are word lists filled into the `{always_tag}` /
`{never_tag}` placeholders of the instruction (joined with ", "; an empty list renders as "אין").
The output is validated: with the tags stripped it must equal the input exactly (only trailing
whitespace / line endings may differ) and every tag must be well formed. There is no untagged fallback: if the LLM call fails or the output
is invalid, that siman fails (no transcript, no TTS) and the run continues with the next siman. Tags are passed unchanged to every TTS engine
(ElevenLabs v3/v4 honors them; Google TTS ignores them silently), and TTS chunking never splits a tag.

Standalone (Windows PowerShell, from the repo root):

```powershell
# Tag an existing transcript (writes <input>_tagged.txt and <input>_tagged_tags.txt)
.venv\Scripts\python scripts\tag_pronunciation.py output\<file>_transcript.txt --apply-nikkud
# Synthesize any text file with the TTS engine configured in config.yaml
.venv\Scripts\python scripts\tts_from_file.py output\<file>_transcript_tagged.txt
```

`scripts/tag_pronunciation.py` likewise writes nothing and exits with code 1 if tagging fails.

### Experiment: combined polish + tagging vs two stages

Compare cost and quality of one LLM call (polish + tags) against the pipeline's two calls.
Each script prints prompt / output / thinking tokens and an estimated USD cost, using the per-1M
prices in `experiment_pricing:` in `config.yaml` (add an entry for any other model you use).

```powershell
# One call: polish + tags (writes <draft>_combined_tagged.txt and ..._tags.txt)
.venv\Scripts\python scripts\polish_and_tag_combined.py output\drafts\<draft>.txt output\relations\<relations>.txt --apply-nikkud
# Two calls: polish only, then tag the result
.venv\Scripts\python scripts\polish_draft.py output\drafts\<draft>.txt output\relations\<relations>.txt
.venv\Scripts\python scripts\tag_pronunciation.py output\drafts\<draft>_polished.txt --apply-nikkud
```

The combined system instruction is `polishing_instruction`, then
`experiment_combined_polish_tag.addendum_header`, then `pronunciation_instruction` without its first
paragraph (the "return the script exactly as is" lines, which conflict with polishing).
Phoneme tags in the combined output must be well formed, or the script exits 1.

### Run summary and exit code

A failure in Stage 3 for one siman (polishing, pronunciation tagging, nikkud/transcript save, TTS)
fails only that siman; the remaining simanim still run. At the end, `main.py` logs a summary and
exits with code 1 if any siman failed (0 otherwise):

```
===== Run summary =====
Succeeded (2/3): 94, 96
Failed (1/3): 95
  - Siman 95 | stage: Stage 3.4 (Pronunciation tagging) | reason: pronunciation tagging output rejected: text changed outside tags: ...
```

Setup errors and Stage 1 / Stage 2 failures still abort the whole run immediately (exit 1),
since the relations map needs every context draft.

Output files are saved to `./output/` with timestamps preserved.

---

## CLI Usage

```
python main.py <SIMANIM> [OPTIONS]
```

### Basic Usage

```bash
# Single siman
python main.py 94

# Multiple simanim and ranges
python main.py 94,95-97,100

# Add broader context for cross-references
python main.py 94 --context-range 87-111
```

### Running Options

| Option | Description |
|--------|-------------|
| `--stage-1-only` | Extract only (skip Stages 2-3) |
| `--skip-tts` | Generate transcripts only, no audio |
| `--limit-lessons N` | Process only first N simanim (debug mode) |
| `--overwrite-cache` | Regenerate cached drafts and relations |
| `--relations-file PATH` | Use pre-existing relations map |

### Batch Mode (Cost-Saving)

```bash
# Submit batch job (50% cheaper, bypasses rate limits)
python main.py 94 --context-range 87-111 --batch

# Retrieve results when complete
python main.py 94 --retrieve-batch <batch_id>
```

With `--batch`, every LLM stage runs as batch jobs with polling, including pronunciation tagging
(Stage 3.4): one tagging job per siman that polished successfully, using the `pronunciation:` engine,
model, temperature and `thinking_level` (which may differ from the generator engine). Each result
goes through the same validation as sync mode; a failed job, empty result or rejected output fails
only that siman (stage "Stage 3.4 (Pronunciation tagging)" in the run summary). The run stops at
startup if `pronunciation.engine` does not support batch.

---

## Configuration

### Generator Engine

```yaml
generator:
  engine: "gemini"  # or "openai"
  
  gemini:
    model_name: "gemini-3.1-pro-preview"
    temperature: 0.3
    
  openai:
    model_name: "gpt-5.2"
    temperature: 1.0
    service_tier: "auto"
```

### TTS Engine

```yaml
tts:
  engine: "gemini"  # "elevenlabs", "google", "openai", or "gemini"
  chunk_gap_paragraph_ms: 700  # silence between TTS chunks split at a paragraph (0 = none)
  chunk_gap_sentence_ms: 300   # silence between TTS chunks split mid-paragraph (0 = none)
  
  gemini:
    voice_name: "Achird"  # Achird, Puck, Charon, Kore, Fenrir, Aoede
    chunk_size: 2000
    
  elevenlabs:
    voice_id: "lJylpTXX0sNdqq5EUv4M"
    chunk_size: 8000      # eleven_v4 allows 10,000 chars/request; headroom for <phoneme> markup
    context_chars: 500    # previous_text / next_text length for request stitching
    
  google:
    voice_name: "he-IL-Chirp3-HD-Achird"
    chunk_size: 2000      # Google's limit is 5000 BYTES; Hebrew = 2 bytes/char, so max 2500
    
  openai:
    voice: "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    chunk_size: 4000      # OpenAI hard limit is 4096
```

`tts.<engine>.chunk_size` (max characters per TTS request) is required for the selected engine and
must be a positive integer; there are no defaults in code. Chunks are split at paragraph / sentence
boundaries and never inside a `<phoneme>` tag.

**ElevenLabs request stitching:** every chunk is sent with `previous_text` (the last
`context_chars` characters of the preceding chunk) and `next_text` (the first `context_chars`
characters of the following chunk), with phoneme tags stripped and cut at a word boundary, so
prosody stays continuous across chunk boundaries. The first chunk has no `previous_text`, the last
has no `next_text`, and a single-chunk text sends neither.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ELEVENLABS_API_KEY` | ElevenLabs API key |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google service account JSON |

---

## Output Files

```
output/
├── {section}_Siman_{N}_{model}_transcript.txt      # Latest transcript
├── {section}_Siman_{N}_{model}_transcript_YYYYMMDD_HHMMSS.txt  # Timestamped history
├── {section}_Siman_{N}_{model}_pronunciation.txt       # Tagged script (before nikkud), if enabled
├── {section}_Siman_{N}_{model}_pronunciation_tags.txt  # word<TAB>ipa per tagged word, if enabled
├── {section}_Siman_{N}_{model}.mp3               # Audio file
├── drafts/                                         # Stage 1 cached drafts
│   └── {section}_Siman_{N}_{model}_draft.txt
└── relations/                                      # Stage 2 cached relations
    └── {section}_Relations_{range}_{model}.txt
```

- **Latest files**: Overwritten on each run
- **Timestamped files**: Preserve version history

---

## Testing

```bash
# Run the test suite
python -m pytest tests/ -v
```

---

## License

MIT License - see LICENSE file for details.

---

## Support

For issues or questions, please open an issue on GitHub.