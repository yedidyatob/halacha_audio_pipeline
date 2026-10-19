"""Shared fakes for Gemini generator tests: a patched client and real SDK stream chunks."""
from unittest.mock import patch
from google.genai import types
from pipeline.generator import GeminiScriptGenerator

PRICE = {"input_per_million": 2.0, "output_per_million": 12.0,
         "long_context": {"threshold_tokens": 200000, "input_per_million": 4.0, "output_per_million": 18.0}}


def make_gemini_generator(thinking_dir, model_name="fake-model", temperature=0.3, ssl_verify=True, pricing=PRICE):
    """(generator, mock client); the client is a MagicMock standing in for genai.Client()."""
    with patch("pipeline.generator.genai.Client") as client_cls:
        gen = GeminiScriptGenerator(api_key="fake-key", model_name=model_name, temperature=temperature,
                                    ssl_verify=ssl_verify, pricing=pricing, thinking_dir=str(thinking_dir))
    return gen, client_cls.return_value


def chunk(*parts, finish_reason=None, finish_message=None, usage=None, prompt_feedback=None):
    """One stream chunk. parts: ("thought", text) or ("text", text)."""
    candidates = None
    if parts or finish_reason or finish_message:
        candidates = [types.Candidate(
            content=types.Content(role="model", parts=[
                types.Part(text=text, thought=True if kind == "thought" else None) for kind, text in parts
            ]) if parts else None,
            finish_reason=finish_reason, finish_message=finish_message,
        )]
    return types.GenerateContentResponse(candidates=candidates, usage_metadata=usage,
                                         prompt_feedback=prompt_feedback)


def usage(prompt=1000, output=200, thoughts=300):
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt, candidates_token_count=output, thoughts_token_count=thoughts,
        total_token_count=prompt + output + thoughts)


def answer_stream(text="answer", thought="thinking..."):
    """A normal stream: a thought chunk, the answer in two chunks, STOP + usage on the last one."""
    half = len(text) // 2
    return [chunk(("thought", thought)), chunk(("text", text[:half])),
            chunk(("text", text[half:]), finish_reason=types.FinishReason.STOP, usage=usage())]
