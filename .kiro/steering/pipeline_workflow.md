# Pipeline Workflow: Execution Order & Caching

## The 4 Stages

1. **Stage 1 - Data Extraction**: Generate structured draft from Sefaria
2. **Stage 2 - Relations Analysis**: Map cross-Siman connections
3. **Stage 3 - Script Polishing**: Convert to TTS-optimized monologue
4. **Audio Synthesis**: Generate mp3 via TTS engine

## Cache Behavior

- Drafts are cached in `cache/` as `{section}_Siman_{N}_{model}_draft.txt`
- Relations maps are cached in `cache/` as `{section}_Relations_{range}_{model}.txt`
- Transcripts and audio are saved to `output/` with versioned timestamps
- Cache is automatically reused unless `--overwrite-cache` is passed

## Execution Flags

| Flag | Behavior |
|------|----------|
| `--stage-1-only` | Extract only, skip relations and polishing |
| `--skip-tts` | Generate transcripts only, skip audio |
| `--context-range X-Y` | Use wider context for cross-references |
| `--batch` | Use Gemini/OpenAI batch API (asynchronous) |
| `--limit-lessons N` | Debug: generate only first N Simanim |
| `--relations-file <path>` | Use pre-existing relations map |

## When Stages Are Skipped

| Cached? | Relations File | Action |
|---------|----------------|--------|
| All drafts exist | Exists | Skip Stages 1 & 2 |
| All drafts exist | Missing | Generate Stage 2 only |
| Some drafts missing | Any | Generate missing drafts (Stage 1) |

## Output File Naming

- **Latest**: `Yoreh_Deah_Siman_{N}_{model}_transcript.txt` / `.mp3`
- **History**: `Yoreh_Deah_Siman_{N}_{model}_transcript_YYYYMMDD_HHMMSS.txt` / `.mp3`

## Cost Reporting

Always report estimated API costs at end of execution.