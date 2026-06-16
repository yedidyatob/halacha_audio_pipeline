# Halacha Audio Lesson Generation Pipeline

An end-to-end, production-grade Python pipeline to download classical Jewish Halachic source texts from the Sefaria API, generate custom Text-to-Speech (TTS) optimized Hebrew monologue scripts using Gemini (google-genai) or OpenAI (including reasoning models like `o1` and `o3-mini`), and synthesize them to high-quality audio files (.mp3) using ElevenLabs, OpenAI TTS, or Google Cloud TTS.

This pipeline is designed specifically for advanced students preparing for the Rabbinical Ordination exams, producing single-voice monologue lessons that sound like an authentic rabbi speaking directly to you, incorporating spoken transitions, definitions, and cross-references.

---

## Project Structure

```
halacha_audio_pipeline/
│
├── config.example.yaml       # Template configuration (credentials, TTS voice selection, prompts)
├── config.yaml               # Application configuration (git-ignored, local sensitive keys)
├── requirements.txt          # Python dependencies
├── main.py                   # CLI Orchestrator entry point
├── .gitignore                # Excludes credentials, caches, outputs, and JSON dumps
│
├── pipeline/
│   ├── __init__.py
│   ├── config.py             # Configuration loader & validator
│   ├── gematria.py           # Gematria converter (e.g. 94 -> צ"ד)
│   ├── input_parser.py       # Commas and ranges string parser (e.g. "94,95-97")
│   ├── extractor.py          # Sefaria v3/v1 API client & HTML clean parser
│   ├── generator.py          # Gemini & OpenAI API wrappers (Master Reference Strategy)
│   ├── tts.py                # Abstract TTS interface (ElevenLabs, OpenAI, Google Cloud TTS)
│   └── logger.py             # Standard logging configuration
│
└── tests/                    # Complete pytest suite (38 passing tests)
    ├── __init__.py
    ├── test_extractor.py
    ├── test_gematria.py
    ├── test_generator.py
    ├── test_input_parser.py
    └── test_tts.py
```

---

## Setup Instructions

### 1. Prerequisite Python Environment
Use the pre-installed virtual environment at `..\v` or activate a local virtual environment:
```bash
# To run commands with the existing virtual environment:
..\v\Scripts\python.exe main.py <arguments>
```

To initialize your own:
```bash
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Local Environment
1. Copy `config.example.yaml` to `config.yaml`:
   ```bash
   cp config.example.yaml config.yaml
   ```
2. Open `config.yaml` and fill in your API credentials:
   ```yaml
   api_keys:
     gemini_api_key: "YOUR_GEMINI_API_KEY"
     openai_api_key: "YOUR_OPENAI_API_KEY"         # Required if using OpenAI generator or OpenAI TTS
     elevenlabs_api_key: "YOUR_ELEVENLABS_API_KEY" # Required if using ElevenLabs
     google_tts_credentials: "C:/path/to/credentials.json" # Required if using Google Cloud TTS
   ```

*Note: You can also set these via standard environment variables (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`), which the pipeline will automatically detect.*

---

## How to Run the Pipeline

Run the pipeline from the command line by supplying the target Simanim.

### 1. Standard Synchronous Run
To extract, generate, and synthesize lessons for Simanim 94, 95, 96, and 97:
```bash
..\v\Scripts\python.exe main.py 94,95-97
```

### 2. Broad Context Range
To target a specific Siman (e.g., 94) but feed the model the context of a wider range (e.g., 87-111) to allow it to construct cross-references:
```bash
..\v\Scripts\python.exe main.py 94 --context-range 87-111
```

### 3. OpenAI Batch API (Cost-saving & Rate Limit Bypass)
For very large context ranges, run asynchronously using the OpenAI Batch API (50% cheaper, bypasses tokens-per-minute limits):
```bash
# Submit the batch job:
..\v\Scripts\python.exe main.py 94 --context-range 87-111 --batch

# Retrieve and polish the results once completed (typically takes a few minutes):
..\v\Scripts\python.exe main.py 94 --retrieve-batch <batch_id>
```

### 4. Skip TTS (Text Only / Transcript Optimization)
To download texts and generate scripts without running the TTS synthesis (saves credits while testing script content):
```bash
..\v\Scripts\python.exe main.py 94 --context-range 93-95 --skip-tts
```

### 5. Debug / Limit Mode
To compile a large range of context (e.g. 91-97) but limit generation and audio synthesis to only the first Siman (91) for testing:
```bash
..\v\Scripts\python.exe main.py 91-97 --limit-lessons 1
```

---

## Versioning & Output Files

Transcripts and audio lessons are saved in the `./output` directory.
To prevent overwriting previous versions during prompt optimization, the pipeline saves two files:
1. `Yoreh_Deah_Siman_X_transcript.txt` (and `.mp3`) — The latest copy, overwritten on each run.
2. `Yoreh_Deah_Siman_X_transcript_YYYYMMDD_HHMMSS.txt` (and `.mp3`) — A timestamped history file that preserves the version history.

If the latest `.mp3` copy is locked (e.g. playing in your media player), the script will output a warning and continue, leaving the timestamped file fully saved.

---

## Running Tests

To run the automated test suite and ensure all components are working:
```bash
..\v\Scripts\python.exe -m pytest tests/
```
