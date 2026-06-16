# Halacha Audio Lesson Generation Pipeline

An end-to-end, production-grade Python pipeline to download classical Jewish Halachic source texts from the Sefaria API, generate custom Text-to-Speech (TTS) optimized Hebrew monologue scripts using Gemini 1.5 Pro, and synthesize them to high-quality audio files (.mp3) using ElevenLabs or Google Cloud TTS.

This pipeline is designed specifically for advanced students preparing for the Rabbinical Ordination exams, producing single-voice monologue lessons that sound like an authentic rabbi speaking directly to you, incorporating spoken transitions, definitions, and cross-references.

## Project Structure

```
halacha_audio_pipeline/
│
├── config.yaml               # Application configuration (credentials, TTS voice selection, prompts)
├── requirements.txt          # Python dependencies
├── main.py                   # CLI Orchestrator entry point
│
├── pipeline/
│   ├── __init__.py
│   ├── config.py             # Configuration loader & validator
│   ├── gematria.py           # Gematria converter (e.g. 94 -> צ"ד)
│   ├── input_parser.py       # Commas and ranges string parser (e.g. "94,95-97")
│   ├── extractor.py          # Sefaria v3/v1 API client & HTML clean parser
│   ├── generator.py          # Gemini API wrapper (Master Reference Strategy)
│   ├── tts.py                # Abstract TTS interface (ElevenLabs & Google Cloud TTS)
│   └── logger.py             # Standard logging configuration
│
└── tests/                    # Complete pytest suite
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

### 2. Enter API Keys & Configuration
Open [config.yaml](file:///C:/Users/yedidyat/.gemini/antigravity/scratch/halacha_audio_pipeline/config.yaml) and fill in your API credentials:

```yaml
api_keys:
  gemini_api_key: "YOUR_GEMINI_API_KEY"
  elevenlabs_api_key: "YOUR_ELEVENLABS_API_KEY" # Optional, if using ElevenLabs
  google_tts_credentials: "C:/path/to/credentials.json" # Optional, if using Google TTS
```

*Note: You can also set these via standard environment variables (`GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`), which the pipeline will automatically detect.*

---

## How to Run the Pipeline

Run the pipeline from the command line by supplying the target Simanim.

### Standard Run
To extract, generate, and synthesize lessons for Simanim 94, 95, 96, and 97:
```bash
..\v\Scripts\python.exe main.py 94,95-97
```

### Debug / Limit Mode (Highly Recommended for Testing)
To extract a wide range of Simanim for context (e.g., 91 to 97), but **only generate and synthesize 1 lesson** (e.g., for Siman 91) to save API credits:
```bash
..\v\Scripts\python.exe main.py 91-97 --limit-lessons 1
```

### Skip TTS (Text Only)
To only download texts and write script transcripts to text files, skipping the audio synthesis:
```bash
..\v\Scripts\python.exe main.py 94,95-97 --skip-tts
```

---

## Configuration Customization

- **TTS Engine**: Switch between `elevenlabs` and `google` in [config.yaml](file:///C:/Users/yedidyat/.gemini/antigravity/scratch/halacha_audio_pipeline/config.yaml) under the `tts.engine` property.
- **Voices**: Customize ElevenLabs voice ID or Google Neural2 voice name (`he-IL-Neural2-M` or `he-IL-Neural2-F`).
- **Prompting Persona**: Modify the Hebrew prompt in [config.yaml](file:///C:/Users/yedidyat/.gemini/antigravity/scratch/halacha_audio_pipeline/config.yaml) under `gemini.system_instruction`.

---

## Running Tests

To run the automated test suite and ensure all components are working:
```bash
..\v\Scripts\python.exe -m pytest tests/
```
