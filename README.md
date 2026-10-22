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
| **Stage 3** | Generate the polished TTS-ready script, vocalizing only the words the TTS engine could misread (one LLM call per siman, see below) |
| **Stage 3.1** | Dictionary nikkud (`pipeline/apply_nikkud.py`): rabbinic abbreviations (an allowlist also without gershayim, e.g. מהרשל, when the bare form is not an ordinary word), fixed word corrections and letter names in a letter-name context (incl. the respelling הא → הֵי). Matching ignores nikkud, so the dictionary's nikkud always replaces the model's |
| **Stage 3.2** | Save the transcript and its nikkud review file, then TTS |

### Mechaber vs. Rama (Stage 1)

Sefaria prints every Rama gloss on the Shulchan Arukh in `<small>`, both the glosses that open with
הגה and the short inline ones inside the Mechaber's text. Before HTML is stripped, the extractor turns
each top-level `<small>` block into `[רמ"א] ... [/רמ"א]` (a leading הגה is dropped); all other
Shulchan Arukh text is the Mechaber's, including text that continues after a gloss. Other works are
cleaned as before. Rule 6 of `system_instruction` tells the LLM how to read the labels.

### Nikkud (Stages 3 and 3.1)

The Stage 3 polish call also adds the nikkud: `polishing_instruction` in `config.yaml` ends with a
nikkud addendum (`===== תוספת: ניקוד לצורך הגייה =====`) telling the model to vocalize only the words a
reader could read two ways (עובר → עֻבָּר, פת, סופר, letter names used as letters), with no markup. It
uses the polish's model, temperature (`generator.polish_temperature`), thinking and retries, in sync
and batch mode; there is no separate nikkud LLM call.
`pronunciation.always_vocalize` / `pronunciation.never_vocalize` are word lists filled into the
`{always_vocalize}` / `{never_vocalize}` placeholders of `polishing_instruction` (joined with ", "; an
empty list renders as "אין"). An `always_vocalize` entry may also give the word's readings, one or more
(`word` + `readings: [{nikkud, meaning}, ...]`), rendered as
`למה (לְמָה = אל מה; לָמָּה = מדוע)`; at startup each reading must have a meaning, and its
nikkud must be the word's letters (ו / י may be added or removed except as the first letter).

The dictionary stage (3.1) then runs on the vocalized polish. Its matching ignores nikkud, on the
word and on its prefix letters (ו, ה, ב, כ, ל, מ, ש and their combinations), so הָרַמְבַּ"ם, וּבִכְחָל and
סימן הֵא are found like their plain forms. The dictionary always wins: the matched word gets the
dictionary's nikkud (or respelling: הא / הֵא → הֵי). The prefix is kept exactly as the model wrote it,
including its nikkud (וּבִכְחָל → וּבִכְּחָל): the entry vocalizes only the word, and a prefix's vowel
depends on the word that follows, which the model saw. Words not in the dictionary keep the model's
nikkud, and a partial-word match is never touched.

The review file (`..._pronunciation_changes.txt`, saved with every transcript) has two sections:
- the vocalized words of the final transcript: a header line, then one tab-separated line per
  distinct plain → vocalized pair: `varies`, `plain`, `vocalized`, `count`, `first_line` (in the
  transcript) and `context` (up to 5 words on each side of the first occurrence, the word in
  brackets, e.g. `מה דינו של [כְּחָל] של בהמה`). Words vocalized in more than one way (`varies` =
  `yes`) come first, each variant on its own line; then the rest by count.
- after a blank line, `dictionary overrides (the model's nikkud was replaced by the dictionary's)`,
  a header line and one line per place where the dictionary replaced a word the model had
  vocalized differently: `model`, `dictionary`, `line`, `context` (the dictionary's form in brackets).

Standalone (Windows PowerShell, from the repo root):

```powershell
# Synthesize any text file with the TTS engine configured in config.yaml
.venv\Scripts\python scripts\tts_from_file.py output\<file>_transcript.txt
```

### Stage 3 polish cache

The Stage 3 polish (`{section}_Siman_{N}_{model}_polish.txt` in `output_dir`) is also a cache, in
sync and batch mode: when it exists for a target siman, the run loads it instead of calling the LLM
(logged as `Loaded cached Stage 3 polish for Siman N: <path> (<n> chars).`) and Stage 3.1 /
transcript / TTS run as usual; in batch mode only the other simanim get polish jobs, and none are
submitted when every polish is cached. An empty (whitespace-only) cached polish counts as missing
and is regenerated, with a warning. `--overwrite-cache` regenerates it. The cached polish
is the vocalized polish, before the dictionary nikkud, so after editing the dictionary a rerun with
the cache calls no LLM; after editing `polishing_instruction` or `pronunciation.always_vocalize`, use
`--overwrite-cache` to polish again.

### Run summary and exit code

A failure in Stage 3 for one siman (polishing, dictionary nikkud, transcript save, TTS)
fails only that siman; the remaining simanim still run. At the end, `main.py` logs a summary and
exits with code 1 if any siman failed (0 otherwise):

```
===== Run summary =====
Succeeded (2/3): 94, 96
Failed (1/3): 95
  - Siman 95 | stage: Stage 3 (Polishing and nikkud) | reason: batch job failed (job_id: ...) (after 2 attempt(s))
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
- which simanim go through Stage 3 / 3.1 / TTS, with the transcript and audio paths

It then exits with code 0. Nothing is written and no API, LLM, TTS or Sefaria call is made;
only the local cache files are read.

### Batch Mode (Cost-Saving)

```bash
# Submit batch job (50% cheaper, bypasses rate limits)
python main.py 94 --context-range 87-111 --batch

# Retrieve results when complete
python main.py 94 --retrieve-batch <batch_id>
```

With `--batch`, every LLM stage runs as batch jobs with polling, including the Stage 3 polish and
nikkud (one job per siman).

Failures and retries work the same way in every LLM stage. A failed attempt is a failed call in sync
mode; in batch mode a failed submit, a job the API reports as failed / cancelled / expired, or a failed
request inside a succeeded job (logged
with the API's error code and message, e.g. `error code 1 (CANCELLED): The operation was cancelled.`,
or a blocked / cut-off / empty response named by its block or finish reason). Every failed attempt is
logged as a warning. Retries run in rounds: each round retries all the items of the stage that have
failed so far, in sync mode as new sync calls and in batch mode as new batch jobs that are polled
again (batch mode never falls back to sync calls). Stages 1-3 allow up to `generator.max_retries`
rounds.
When an item still fails after its retries:
- Stage 1 (drafts): every successful draft is already saved to the cache (each one as soon as it
  arrives). The run then aborts, and a rerun only regenerates the missing drafts.
- Stage 2 (relations map): the run aborts.
- Stage 3: only that siman fails, and it appears in the end-of-run summary.

A failed status check or result fetch while polling (network / API error) says nothing about the
job, which may still be running or already done. It is logged as a warning and simply tried again
at the next poll (`--batch-poll-interval`); the job is never marked failed or resubmitted for it.
Each result is fetched (and a Stage 1 draft saved) as soon as its job completes. If
`--batch-timeout` expires first, the run stops with an error listing every unresolved job id per
siman and why it is unresolved (still pending, status check failed, or result fetch failed), so the
results can be fetched later.

---

## Configuration

Every setting comes from `config.yaml`; there are no defaults in code, and a missing or invalid
key stops the run at startup with a message naming it. This includes `generator.engine`,
`tts.engine` and the `sefaria:` block:

```yaml
sefaria:
  base_url: "https://www.sefaria.org/api"
  timeout_seconds: 15   # per request
  retries: 3            # attempts per request
  ssl_verify: false
```

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

Every generation setting (model names, temperatures, `service_tier`, `polish_temperature`, and the
`experiments.gemini_native_audio` section) is required in `config.yaml`; there are no
code defaults, and a missing or invalid value is a startup error. `generator.polish_temperature` is the
temperature of the Stage 3 polish calls (sync and batch).

`generator.max_retries` is the number of retries after a failed attempt in Stages 1-3 (API error,
failed batch job or request, empty, blocked or cut-off response; 0 = no retry): new calls in sync
mode, new batch jobs in batch mode.

`generator.draft_thinking_level` (`low`/`medium`/`high`, required for engine `gemini`, ignored for
`openai`) is the Gemini thinking level of the Stage 1 draft calls, sent in both sync and batch mode.

### Gemini calls: streaming and thinking files

Every synchronous Gemini text call (Stages 1-3) goes through one path, `GeminiScriptGenerator._generate`: it streams the
response (`generate_content_stream`) with `thinking_config.include_thoughts` on, so bytes keep
arriving while the model thinks. A long non-streaming call sends nothing until it finishes, and an
idle connection can be cut by a proxy or front end (`Server disconnected without sending a
response` after a few minutes of thinking). Block / finish-reason checks are the same as for batch
results; usage is read from the final chunk.

Only the answer text is returned, saved and passed on. The thought summaries of the Stage 1-3 calls
are written to `directories.thinking_dir`, one file per call named
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
`threshold_tokens`. Every Gemini model used for a text call (the generator's
`model_name`) needs an entry; a missing one is a startup error.

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
    pitch: 0              # Cloud TTS audioConfig.pitch (-20 to 20)
    speaking_rate: 1      # Cloud TTS audioConfig.speakingRate (0.25 to 2.0)
    language_code: "he-il"
    
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
├── {section}_Siman_{N}_{model}_polish.txt          # Stage 3 polished and vocalized text (before the dictionary nikkud)
├── {section}_Siman_{N}_{model}_transcript.txt      # Latest transcript
├── {section}_Siman_{N}_{model}_transcript_YYYYMMDD_HHMMSS.txt  # Timestamped history
├── {section}_Siman_{N}_{model}_pronunciation_changes.txt  # nikkud review: vocalized words and dictionary overrides (see Nikkud)
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