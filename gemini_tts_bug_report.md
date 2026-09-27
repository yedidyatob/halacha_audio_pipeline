# Gemini 3.1 Flash TTS Preview - Bug Report

## Issue Summary
When generating Text-to-Speech (TTS) for Hebrew text using the `gemini-3.1-flash-tts-preview` model, the API enters an infinite generation loop. This results in two critical failures:
1. **Severe Billing Anomalies:** A short 400-character string (approx. 30 seconds of speech) generates upwards of 250,000+ output audio tokens (over $5.00 USD) as the model hallucinates or endlessly loops audio frames.
2. **Endpoint Timeouts:** The massive, runaway audio payload causes the HTTPS connection to exceed the standard 120-second timeout, resulting in a dropped connection (`ReadTimeout`) before any audio data is successfully returned to the client, despite the user being billed for the generation.

## Environment & Configuration
* **Endpoint:** `generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-tts-preview:generateContent`
* **SDK:** `google-genai` (Python) and raw REST API (`requests`)
* **Voice Configuration:** `Achird` (PrebuiltVoiceConfig)
* **Language:** Hebrew
* **Response Modality:** `["AUDIO"]`
* **Input Text Length:** ~400 characters (approx. 70 words).

## Steps to Reproduce
1. Construct a standard `generateContent` request for `gemini-3.1-flash-tts-preview`.
2. Set the `responseModalities` to `["AUDIO"]`.
3. Provide a prompt containing Hebrew text, for example:
   > "מכאן אנו עוברים לסעיף זין, ועוברים מנפילה פסיבית לפעולה אקטיבית של חיתוך. המחבר פוסק שבשר רותח שחתכו בסכין חלבית, כל החתיכה אסורה..."
4. Submit the request.

## Expected Behavior
The API should synthesize the 400-character Hebrew text into approximately 30-40 seconds of PCM/audio data, generating roughly 1,000 - 2,000 audio output tokens. The connection should resolve cleanly within a few seconds.

## Actual Behavior
* The server accepts the request and begins generation but hangs indefinitely.
* The connection times out after 120+ seconds (`requests.exceptions.ReadTimeout`).
* A check of the Google AI Studio / GCP Billing dashboard immediately after the timeout reveals that a single request consumed **> 250,000 output tokens**.
* The user is charged for the massive token output without receiving any usable audio payload.

## Hypothesis
The preview TTS model (`Achird` voice profile) appears to lack native phoneme mapping or robust failure handling for Hebrew characters. Instead of returning a `400 Bad Request` or a clean fallback, the model hallucinates or enters a runaway state, generating endless frames of silence or garbled audio until the connection is forcefully closed by timeout mechanisms. The backend successfully bills for every generated token in this runaway loop.
