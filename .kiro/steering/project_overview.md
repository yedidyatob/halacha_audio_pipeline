# Project Overview: Halacha Audio Pipeline

This is a production-grade Python pipeline for generating Hebrew-language audio lessons from classical Jewish Halachic texts.

## What It Does

The pipeline performs end-to-end processing:

1. **Data Extraction** (Stage 1): Extracts structured Halachic content from Sefaria API for specific Simanim (sections)
2. **Cross-Relations Analysis** (Stage 2): Analyzes thematic connections between multiple Simanim
3. **Script Polishing** (Stage 3): Converts structured data into a TTS-optimized Hebrew monologue script
4. **Audio Synthesis**: Generates high-quality audio using TTS engines (ElevenLabs, OpenAI, Google Cloud, or Gemini)

## Target Audience

Advanced students preparing for Rabbinical Ordination exams. The output sounds like an authentic rabbi speaking directly to the listener.

## Core Design Principles

- **100% Information Preservation**: No content summarization or omission is permitted during script polishing
- **Hebrew-First**: All prompts and outputs are in Hebrew (with specific TTS pronunciation rules)
- **Production-Grade**: Logging, caching, versioning, and batch API support for scalability

## Project Structure

```
halacha_audio_pipeline/
├── main.py                   # CLI entry point (orchestrator)
├── config.yaml               # Configuration (API keys, models, prompts)
├── pipeline/                 # Core modules
│   ├── config.py            # Config loader & validator
│   ├── extractor.py         # Sefaria API client
│   ├── generator.py         # Gemini/OpenAI script generators
│   ├── tts.py               # TTS engine abstraction
│   ├── evaluator.py         # Heuristic quality checks
│   ├── gematria.py          # Hebrew numeral converter
│   └── input_parser.py      # Simanim string parser
└── output/                   # Generated transcripts & audio
```

## Key Constraints

- DO NOT run the script without explicit user request
- Always report API costs at the end of any execution
- Cache intermediate results to avoid unnecessary API calls
- Version all outputs with timestamps to preserve history
