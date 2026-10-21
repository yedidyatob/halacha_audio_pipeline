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

Re-run this after pulling: an existing venv is not upgraded automatically. ElevenLabs request stitching
needs `elevenlabs>=1.3.0`; an older SDK fails with `unexpected keyword argument 'next_text'`.

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
| **Stage 1** | Extract Halachic content from Sefaria API (Rama glosses in the Shulchan Arukh are labeled `[רמ"א] ... [/רמ"א]`, see below) |
| **Stage 2** | Analyze cross-simanim relations for context |
| **Stage 3** | Generate polished TTS-ready script & synthesize audio |
| **Stage 3.4** | Dictionary nikkud (`pipeline/apply_nikkud.py`): rabbinic abbreviations (an allowlist also without gershayim, e.g. מהרשל, when the bare form is not an ordinary word), fixed word corrections and letter names in a letter-name context (incl. the respelling הא → הֵי); never touches already-vocalized words |
| **Stage 3.5** | Pronunciation nikkud (optional, `pronunciation.enabled`): an LLM adds nikkud only to context-ambiguous words (עובר → עֻבָּר, פת, סופר, letter names used as letters). Plain text, no markup |
| **Stage 3.6** | Save the transcript, then TTS |

### Mechaber vs. Rama (Stage 1)

Sefaria prints every Rama gloss on the Shulchan Arukh in `<small>`, both the glosses that open with
הגה and the short inline ones inside the Mechaber's text. Before HTML is stripped, the extractor turns
each top-level `<small>` block into `[רמ"א] ... [/רמ"א]` (a leading הגה is dropped); all other
Shulchan Arukh text is the Mechaber's, including text that continues after a gloss. Other works are
cleaned as before. Rule 6 of `system_instruction` tells the LLM how to read the labels.

### Pronunciation nikkud (Stage 3.5)

Configured in `config.yaml` (`pronunciation:` section + `pronunciation_instruction`; no code defaults).
`pronunciation.thinking_level` (`low`/`medium`/`high`) sets the Gemini thinking level for this call
only; it is required when `pronunciation.engine` is `gemini` and ignored for `openai`.
`pronunciation.max_retries` is the number of retries after a failed attempt (LLM error, empty or
rejected output; 0 = no retry; batch mode retries as new batch jobs, see Batch Mode); each failed
attempt is logged as a warning with its reason.
`pronunciation.always_vocalize` / `pronunciation.never_vocalize` are word lists filled into the
`{always_vocalize}` / `{never_vocalize}` placeholders of the instruction (joined with ", "; an empty
list renders as "אין"). An `always_vocalize` entry may also give the word's readings, one or more
(`word` + `readings: [{nikkud, meaning}, ...]`), rendered as
`כחל (כְּחָל = עטין של בהמה; כֹּחַל = צבע איפור)`; at startup each reading must have a meaning, and its
nikkud must be the word's letters (ו / י may be added or removed except as the first letter, as in
the validation below).

The changes file (`..._pronunciation_changes.txt`) is the review list of what the LLM vocalized: a
header line, then one tab-separated line per distinct original → vocalized pair:
`varies`, `original`, `vocalized`, `count`, `first_line` (in the transcript) and `context` (up to 5
words on each side of the first occurrence, the word in brackets, e.g. `מה דינו של [כְּחָל] של בהמה`).
Words the LLM vocalized in more than one way (`varies` = `yes`) come first, each variant on its own
line; then the rest by count.

The stage runs after the dictionary nikkud, and the LLM is told to leave already-vocalized words
alone. Validation is strict and word by word: words, whitespace and lines must be unchanged (only
trailing whitespace / line endings may differ); with nikkud removed each word must equal the input
word, except that ו / י may be added or removed for standard voweled spelling (עובר → עֻבָּר,
הקילו → הֵקֵלּוּ), never as the word's first letter; every word that already had nikkud must be
byte-identical; and no markup (`<...>`) may appear. There is no fallback: if every attempt fails,
that siman fails with the last reason (no transcript, no TTS) and the run continues with the next
siman.
Letter names whose spelling must change (הא / הי → הֵי, and the wrong vocalization הֵא → הֵי) are
handled by the dictionary stage (`NO_PREFIX_NIKKUD_DICT`, `LETTER_NAME_MISVOCALIZATIONS`), not by
the LLM; the polish writes letter names without nikkud.

Standalone (Windows PowerShell, from the repo root):

```powershell
# Dictionary nikkud + LLM pronunciation nikkud on any text file
# (writes <input>_vocalized.txt and <input>_vocalized_changes.txt, the changes file described above)
.venv\Scripts\python scripts\vocalize_pronunciation.py output\<file>.txt
# Synthesize any text file with the TTS engine configured in config.yaml
.venv\Scripts\python scripts\tts_from_file.py output\<file>_vocalized.txt
```

`scripts/vocalize_pronunciation.py` retries like the pipeline (`pronunciation.max_retries`), then
writes nothing and exits with code 1 if every attempt fails.

### Stage 3 polish cache

The Stage 3 polish (`{section}_Siman_{N}_{model}_polish.txt` in `output_dir`) is also a cache, in
sync and batch mode: when it exists for a target siman, the run loads it instead of calling the LLM
(logged as `Loaded cached Stage 3 polish for Siman N: <path> (<n> chars).`) and Stages 3.4 / 3.5 /
transcript / TTS run as usual; in batch mode only the other simanim get polish jobs, and none are
submitted when every polish is cached. An empty (whitespace-only) cached polish counts as missing
and is regenerated, with a warning. `--overwrite-cache` regenerates it. For example, after editing
`pronunciation.always_vocalize`, `python main.py 87 --context-range 87-111 --skip-tts` calls the LLM
only for Stage 3.5 (the drafts and the relations map are cached too).

### Run summary and exit code

A failure in Stage 3 for one siman (polishing, dictionary nikkud, pronunciation nikkud, transcript save, TTS)
fails only that siman; the remaining simanim still run. At the end, `main.py` logs a summary and
exits with code 1 if any siman failed (0 otherwise):

```
===== Run summary =====
Succeeded (2/3): 94, 96
Failed (1/3): 95
  - Siman 95 | stage: Stage 3.5 (Pronunciation nikkud) | reason: pronunciation output rejected: text changed (ignoring nikkud): ...
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
| `--overwrite-cache` | Regenerate cached drafts, relations and Stage 3 polish |
| `--relations-file PATH` | Use pre-existing relations map |
| `--dry-run` | Print the run plan and exit (no API / LLM / TTS / Sefaria calls) |

### Dry Run

```bash
python main.py 94 --context-range 87-111 --batch --dry-run
```

`--dry-run` resolves the run exactly like a real run with the same arguments (the same code
path, `resolve_run_plan`, honouring `--overwrite-cache`, `--relations-file`, `--context-range`,
`--limit-lessons`, `--batch`, `--skip-tts` and `--stage-1-only`). It prints:
- whether the relations map is loaded (from `--relations-file` or the cache) or generated, and its path
- which Stage 1 drafts load from the cache and which are requested, with their paths
- per target siman, whether its Stage 3 polish loads from the cache or is generated, with its path
- which simanim go through Stage 3 / 3.5 / TTS, with the transcript and audio paths

It then exits with code 0. Nothing is written and no API, LLM, TTS or Sefaria call is made;
only the local cache files are read.

### Batch Mode (Cost-Saving)

```bash
# Submit batch job (50% cheaper, bypasses rate limits)
python main.py 94 --context-range 87-111 --batch

# Retrieve results when complete
python main.py 94 --retrieve-batch <batch_id>
```

With `--batch`, every LLM stage runs as batch jobs with polling, including pronunciation nikkud
(Stage 3.5): one job per siman that polished successfully, using the `pronunciation:` engine,
model, temperature and `thinking_level` (which may differ from the generator engine). The run stops at
startup if `pronunciation.engine` does not support batch.

Failures and retries work the same way in every LLM stage. A failed attempt is a failed call in sync
mode; in batch mode a failed submit, a job the API reports as failed / cancelled / expired, or a failed
request inside a succeeded job, including a Stage 3.5 output rejected by validation (logged
with the API's error code and message, e.g. `error code 1 (CANCELLED): The operation was cancelled.`,
or a blocked / cut-off / empty response named by its block or finish reason). Every failed attempt is
logged as a warning. Retries run in rounds: each round retries all the items of the stage that have
failed so far, in sync mode as new sync calls and in batch mode as new batch jobs that are polled
again (batch mode never falls back to sync calls). Stages 1-3 allow up to `generator.max_retries`
rounds and Stage 3.5 up to `pronunciation.max_retries`; Stage 3.5 results also go through the same
validation as sync mode.
When an item still fails after its retries:
- Stage 1 (drafts): every successful draft is already saved to the cache (each one as soon as it
  arrives). The run then aborts, and a rerun only regenerates the missing drafts.
- Stage 2 (relations map): the run aborts.
- Stages 3 / 3.5: only that siman fails, and it appears in the end-of-run summary.

A failed status check or result fetch while polling (network / API error) says nothing about the
job, which may still be running or already done. It is logged as a warning and simply tried again
at the next poll (`--batch-poll-interval`); the job is never marked failed or resubmitted for it.
Each result is fetched (and a Stage 1 draft saved) as soon as its job completes. If
`--batch-timeout` expires first, the run stops with an error listing every unresolved job id per
siman and why it is unresolved (still pending, status check failed, or result fetch failed), so the
results can be fetched later.

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

`generator.max_retries` is the number of retries after a failed attempt in Stages 1-3 (API error,
failed batch job or request, empty, blocked or cut-off response; 0 = no retry): new calls in sync
mode, new batch jobs in batch mode.

`generator.draft_thinking_level` (`low`/`medium`/`high`, required for engine `gemini`, ignored for
`openai`) is the Gemini thinking level of the Stage 1 draft calls, sent in both sync and batch mode.

### Gemini calls: streaming and thinking files

Every synchronous Gemini text call (Stages 1-3, pronunciation nikkud in sync mode, and
`enhance_transcript.py`) goes through one path, `GeminiScriptGenerator._generate`: it streams the
response (`generate_content_stream`) with `thinking_config.include_thoughts` on, so bytes keep
arriving while the model thinks. A long non-streaming call sends nothing until it finishes, and an
idle connection can be cut by a proxy or front end (`Server disconnected without sending a
response` after a few minutes of thinking). Block / finish-reason checks are the same as for batch
results; usage is read from the final chunk.

Only the answer text is returned, saved and passed on. The thought summaries of the Stage 1-3 calls
(and of `enhance_transcript.py`) are written to `directories.thinking_dir`, one file per call named
after that call's output plus `_thinking` (e.g. `{section}_Siman_{N}_{model}_draft_thinking.txt`,
`{section}_Relations_{range}_{model}_thinking.txt`, `{section}_Siman_{N}_{model}_polish_thinking.txt`),
with the usual timestamped history copy. They are for reading only and never feed a later stage.
Batch mode is unchanged: batch requests do not ask for thoughts, and any thought part in a batch
response is dropped.

### Cost tracker

Each synchronous Gemini call logs its tokens (input, output, thinking) and estimated cost. Thinking
tokens are billed as output. Prices come from the `gemini_pricing:` section of `config.yaml` (USD per
1M tokens, paid tier, standard / non-batch, from https://ai.google.dev/gemini-api/docs/pricing); an
optional `long_context` tier applies to the whole request when the prompt exceeds
`threshold_tokens`. Every Gemini model used for a text call (generator, pronunciation,
`enhance_transcript.py --model`) needs an entry; a missing one is a startup error.

```yaml
gemini_pricing:
  gemini-3.1-pro-preview:
    input_per_million: 2.00
    output_per_million: 12.00
    long_context: {threshold_tokens: 200000, input_per_million: 4.00, output_per_million: 18.00}
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
    chunk_size: 8000      # eleven_v4 allows 10,000 chars/request
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
boundaries.

**ElevenLabs request stitching:** every chunk is sent with `previous_text` (the last
`context_chars` characters of the preceding chunk) and `next_text` (the first `context_chars`
characters of the following chunk), cut at a word boundary, so
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
├── {section}_Siman_{N}_{model}_polish.txt          # Stage 3 polished text (before nikkud)
├── {section}_Siman_{N}_{model}_transcript.txt      # Latest transcript
├── {section}_Siman_{N}_{model}_transcript_YYYYMMDD_HHMMSS.txt  # Timestamped history
├── {section}_Siman_{N}_{model}_pronunciation_changes.txt  # review list of the words the LLM changed (see Pronunciation nikkud), if enabled
├── {section}_Siman_{N}_{model}.mp3               # Audio file
├── drafts/                                         # Stage 1 cached drafts
│   └── {section}_Siman_{N}_{model}_draft.txt
├── relations/                                      # Stage 2 cached relations
│   └── {section}_Relations_{range}_{model}.txt
└── thinking/                                       # directories.thinking_dir: thought summaries (read-only)
    └── {output base name}_thinking.txt
```

- **Latest files**: Overwritten on each run
- **Timestamped files**: Preserve version history
- Every saved file is logged at INFO, one line per file: `<stage> output for Siman <N> saved: <latest path> (history copy: <timestamped path>)`;
  thinking files as `Thinking summary of <call, e.g. Gemini script generation for Siman N> saved: …`

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