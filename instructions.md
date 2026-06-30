# Halacha Audio Pipeline Instructions

This document provides instructions on how to run the pipeline, how to configure different text-to-speech (TTS) engines, and how to isolate specific stages for development and debugging.

## 1. Running the Script Normally

To run the full end-to-end pipeline (Data Extraction -> Relations Analysis -> Script Polishing -> Audio Synthesis):

```bash
python main.py <SIMANIM>
```

**Example:**
```bash
python main.py 94
python main.py 94,95-97
```
This will process the requested Simanim, cache the intermediate drafts and relations in the `cache/` directory, and output the final transcripts and audio files to the `output/` directory.

## 2. Configuration (`config.yaml` & `.env`)

The pipeline's behavior is controlled by `config.yaml` and environment variables in the `.env` file.

### Environment Variables (.env)
You must provide API keys for the services you intend to use. Create a `.env` file based on `.env.example`:
```env
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

### config.yaml Settings
The `config.yaml` file controls the underlying models, prompts, and TTS engines.

#### Switching TTS Engines (ElevenLabs, OpenAI, Gemini)
To change the TTS provider, locate the `tts` section in `config.yaml` and change the `engine` value:

**For ElevenLabs:**
```yaml
tts:
  engine: "elevenlabs"
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"
    model_id: "eleven_v3"
    # ...
```
*(Make sure `ELEVENLABS_API_KEY` is set in your `.env`)*

**For OpenAI:**
```yaml
tts:
  engine: "openai"
  openai:
    voice: "alloy"
    model: "tts-1"
    # ...
```
*(Make sure `OPENAI_API_KEY` is set in your `.env`)*

You can similarly switch the generation engine (Stage 1 & 2) between Gemini and OpenAI under the `generator:` block by changing `engine: "gemini"` to `engine: "openai"`.

## 3. Dev: Running Specific Stages

The pipeline is divided into stages and utilizes caching. If a stage's output is already cached, it will be skipped automatically unless you pass `--overwrite-cache`. 

Here is how you can isolate specific stages for development:

### Run ONLY Stage 1 (Data Extraction)
To only extract the raw drafts from Sefaria without proceeding further:
```bash
python main.py 94 --stage-1-only
```
This will create the Stage 1 draft in the cache and exit.

### Run ONLY Stage 2 (Relations Analysis)
There is no explicit `--stage-2-only` flag, but you can run up to Stage 2 (and 3) while skipping TTS by using:
```bash
python main.py 94 --skip-tts
```
If Stage 1 is already cached, it will skip straight to generating the Stage 2 relations map and the Stage 3 script, without synthesizing audio.

*Tip: If you want to skip Stage 2 entirely and use a pre-existing relations map, use `--relations-file <path>`.*

### Run ONLY Stage 3 (Polishing / Monologue Generation)
Stage 3 relies on the outputs of Stages 1 and 2. 
If Stage 1 and Stage 2 have already been run (and are cached), you can run Stage 3 (without synthesizing audio) by executing:
```bash
python main.py 94 --skip-tts
```
Because the drafts and relations are cached, Stages 1 and 2 will be skipped, and only Stage 3 will execute.

### Run ONLY TTS (Text-to-Speech)
Similar to Stage 3, if Stages 1 and 2 are already cached, running the normal command will skip straight to Stage 3 and then perform TTS:
```bash
python main.py 94
```
*Note: Stage 3 (polishing) runs quickly and isn't cached as a separate file before TTS, so running this will re-run the fast Stage 3 polish pass and then trigger the TTS engine.*

### Force Regeneration
If you made prompt changes or want to ignore the cache for Stages 1 and 2:
```bash
python main.py 94 --overwrite-cache
```
