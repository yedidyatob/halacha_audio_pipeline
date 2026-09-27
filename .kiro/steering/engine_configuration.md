# Engine Configuration: Generator & TTS

## Generator Engines

### Gemini (Default)
```yaml
generator:
  engine: "gemini"
  gemini:
    model_name: "gemini-3.1-pro-preview"
    temperature: 0.3
```

### OpenAI (Premium Reasoning)
```yaml
generator:
  engine: "openai"
  openai:
    model_name: "gpt-5.2"  # or o1, o3-mini
    temperature: 1.0       # Ignored for o1/o3 models
    service_tier: "auto"   # flex or auto
```

## TTS Engines

### ElevenLabs (Default)
```yaml
tts:
  engine: "elevenlabs"
  elevenlabs:
    voice_id: "UEKYgullGqaF0keqT8Bu"
    model_id: "eleven_v3"
    stability: 0.5
    similarity_boost: 0.75
```

### Google Cloud TTS
```yaml
tts:
  engine: "google"
  google:
    voice_name: "he-IL-Chirp3-HD-Achird"
    language_code: "he-IL"
    speaking_rate: 1.0
    pitch: 0.0
```

### OpenAI TTS
```yaml
tts:
  engine: "openai"
  openai:
    voice: "alloy"
    model: "tts-1"
    speed: 1.0
```

### Gemini TTS
```yaml
tts:
  engine: "gemini"
  gemini:
    model_name: "gemini-3.1-flash-tts-preview"
    voice_name: "Achird"
    temperature: 0.3
```

## Environment Variables

Set in `.env` file:
```env
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/creds.json
```

## Engine Selection Guidelines

- **Generator**: Use Gemini for speed/cost, OpenAI o1/o3 for reasoning quality
- **TTS**: ElevenLabs for natural Hebrew voice, Gemini TTS for integration simplicity