"""Batch results: per-request errors (Gemini / OpenAI) and the shared retry path of Stages 1, 2, 3."""
import json
from unittest.mock import MagicMock, patch, mock_open

import pytest
from google.genai import types

import main as main_mod
from pipeline.generator import BatchCapableGenerator, GeminiScriptGenerator, OpenAIScriptGenerator
from pipeline.retry import call_with_retries


# --- Gemini: InlinedResponse carries `response` OR `error` ---------------------------

def _gemini_with(*inlined):
    with patch("google.genai.Client") as client_cls:
        gen = GeminiScriptGenerator(api_key="k", model_name="gemini-pro", temperature=0.2)
    job = types.BatchJob(name="batches/1", dest=types.BatchJobDestination(inlined_responses=list(inlined)))
    client_cls.return_value.batches.get.return_value = job
    return gen


def _response(*parts, finish=types.FinishReason.STOP, **kwargs):
    content = types.Content(role="model", parts=list(parts)) if parts else None
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content, finish_reason=finish)], **kwargs)


def test_gemini_batch_result_text_without_thought_parts():
    gen = _gemini_with(types.InlinedResponse(response=_response(
        types.Part(text="thinking...", thought=True), types.Part(text="טיוטה "), types.Part(text="מלאה"))))
    assert gen.get_batch_result("batches/1") == "טיוטה מלאה"


def test_gemini_batch_request_error_is_surfaced():
    gen = _gemini_with(types.InlinedResponse(error=types.JobError(code=13, message="Internal error encountered.")))
    with pytest.raises(ValueError, match=r"batch request failed \(job batches/1\): error code 13: Internal error encountered\."):
        gen.get_batch_result("batches/1")


@pytest.mark.parametrize("inlined, reason", [
    (types.InlinedResponse(), "neither a response nor an error"),
    (types.InlinedResponse(response=types.GenerateContentResponse(candidates=[])), "no candidates"),
    (types.InlinedResponse(response=types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY))),
     "prompt blocked .*SAFETY"),
    (types.InlinedResponse(response=_response(types.Part(text="חצי"), finish=types.FinishReason.MAX_TOKENS)),
     r"stopped early \(finish_reason MAX_TOKENS\).*3 chars of partial text discarded"),
    (types.InlinedResponse(response=_response(finish=types.FinishReason.SAFETY)), "stopped early .*SAFETY"),
    (types.InlinedResponse(response=_response(types.Part(text=""))), r"no text \(finish_reason STOP\)"),
])
def test_gemini_batch_result_failures_name_the_reason(inlined, reason):
    with pytest.raises(ValueError, match=reason):
        _gemini_with(inlined).get_batch_result("batches/1")


def test_gemini_batch_job_without_responses():
    with pytest.raises(ValueError, match="has no inlined response"):
        _gemini_with().get_batch_result("batches/1")


# --- OpenAI: failed requests are in the error file -------------------------------------

def _openai_with(output=None, error=None):
    with patch("pipeline.generator.OpenAI") as openai_cls:
        gen = OpenAIScriptGenerator(api_key="k", model_name="gpt")
    client = openai_cls.return_value
    client.batches.retrieve.return_value = MagicMock(output_file_id=output and "out", error_file_id=error and "err")
    files = {"out": output, "err": error}
    client.files.content.side_effect = lambda file_id: MagicMock(text=json.dumps(files[file_id]) + "\n")
    return gen


def _ok(content, finish="stop"):
    return {"custom_id": "c", "response": {"status_code": 200, "body": {
        "choices": [{"message": {"content": content}, "finish_reason": finish}]}}, "error": None}


def test_openai_batch_result_success():
    assert _openai_with(output=_ok("טיוטה")).get_batch_result("b") == "טיוטה"


@pytest.mark.parametrize("output, error, reason", [
    (None, {"custom_id": "c", "response": None, "error": {"code": "server_error", "message": "boom"}},
     r"request failed \(error code server_error: boom\)"),
    (None, {"custom_id": "c", "response": {"status_code": 429, "body": {"error": {"message": "rate"}}}, "error": None},
     r"HTTP 429: rate"),
    (_ok("חצי", finish="length"), None, "stopped early .*length"),
    (_ok(None), None, r"no text \(finish_reason stop\)"),
    ({"custom_id": "c", "response": {"status_code": 200, "body": {"choices": []}}, "error": None}, None, "no choices"),
    (None, None, "without an output or error file"),
])
def test_openai_batch_result_failures_name_the_reason(output, error, reason):
    with pytest.raises(ValueError, match=reason):
        _openai_with(output=output, error=error).get_batch_result("b")


# --- retry helper ------------------------------------------------------------------

def test_call_with_retries_first_attempt_then_calls(caplog):
    calls = []
    def call():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("again")
        return "ok"
    def batch():
        raise ValueError("request failed")
    assert call_with_retries(call, 2, "Siman 94 X", first_attempt=batch) == "ok"
    assert len(calls) == 2
    assert "Siman 94 X attempt 1/3 failed: request failed; retrying" in caplog.text


def test_call_with_retries_raises_last_reason():
    with pytest.raises(RuntimeError, match=r"^last \(after 2 attempt\(s\)\)$"):
        call_with_retries(MagicMock(side_effect=ValueError("last")), 1, "x",
                          first_attempt=MagicMock(side_effect=ValueError("first")))


# --- config ------------------------------------------------------------------------

BASE_YAML = """
halachic_section: "Yoreh De'ah"
tts:
  chunk_gap_paragraph_ms: 0
  chunk_gap_sentence_ms: 0
  elevenlabs:
    chunk_size: 8000
    context_chars: 500
generator:
  engine: "gemini"
"""


def _config(line):
    from pipeline.config import PipelineConfig
    with patch("builtins.open", mock_open(read_data=BASE_YAML + line)), \
         patch("os.path.exists", return_value=True), patch("os.makedirs"):
        return PipelineConfig("config.yaml")


@pytest.mark.parametrize("value", [0, 2])
def test_config_generator_max_retries(value):
    assert _config(f"  max_retries: {value}\n").generator_max_retries == value


def test_config_generator_max_retries_required():
    with pytest.raises(ValueError, match="'max_retries' under 'generator'"):
        _config("").generator_max_retries


@pytest.mark.parametrize("bad", ["-1", "1.5", '"1"', "true"])
def test_config_generator_max_retries_invalid(bad):
    with pytest.raises(ValueError, match="generator.max_retries' must be a non-negative integer"):
        _config(f"  max_retries: {bad}\n").generator_max_retries


def test_real_config_generator_max_retries():
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        assert PipelineConfig("config.yaml").generator_max_retries == 1


# --- Stage 1 / 2 / 3 in main.py -----------------------------------------------------

class FakeGenerator(BatchCapableGenerator):
    """Batch jobs answer via batch(custom_id); sync calls via sync(kind, key) (an Exception is raised)."""

    def __init__(self, batch=lambda cid: f"draft {cid}", sync=lambda kind, key: f"sync {kind} {key}"):
        self.batch, self.sync = batch, sync
        self.model_name, self.temperature = "m", 0.2
        self.sync_calls, self.submitted = [], []

    def _answer(self, value):
        if isinstance(value, Exception):
            raise value
        return value

    def submit_batch(self, system_instruction, user_prompt, temperature, custom_id, cache_dir="./cache",
                     thinking_level=None):
        self.submitted.append(custom_id)
        return custom_id

    def get_batch_status(self, job_id):
        return "completed"

    def get_batch_result(self, job_id):
        return self._answer(self.batch(job_id))

    def _sync(self, kind, key):
        self.sync_calls.append((kind, key))
        return self._answer(self.sync(kind, key))

    def generate_siman_script(self, siman, master_context, system_instruction, commentators_desc):
        return self._sync("draft", siman)

    def analyze_cross_relations(self, drafts, relations_instruction):
        return self._sync("relations", tuple(drafts))

    def polish_siman_script(self, script_text, relations_text, polishing_instruction):
        return self._sync("polish", script_text)

    def _get_generation_user_prompt(self, gematria_siman, master_context, commentators_desc):
        return master_context

    def _get_relations_user_prompt(self, drafts):
        return "rel"

    def _get_polishing_user_prompt(self, draft, relations_text):
        return draft


REQUEST_ERROR = ValueError("Gemini batch request failed (job stage1_siman_95): error code 13: Internal error")


def _cfg(tmp_path, retries):
    return MagicMock(drafts_dir=str(tmp_path), cache_dir=str(tmp_path), section_slug="YD",
                     generator_max_retries=retries, section_metadata={"prompt_commentators_desc": "d"})


def _args(batch=True):
    return MagicMock(batch=batch, batch_poll_interval=0, batch_timeout=60)


def _saved(tmp_path):
    return sorted(int(p.name.split("_")[2]) for p in tmp_path.glob("YD_Siman_*_m_draft.txt"))


def _extractor():
    return MagicMock(compile_simanim_context=lambda simanim, target_simanim: f"context {simanim[0]}")


def test_stage1_batch_request_error_retried_synchronously(tmp_path):
    gen = FakeGenerator(batch=lambda cid: REQUEST_ERROR if cid.endswith("_95") else f"draft {cid}")
    drafts = main_mod.generate_drafts([94, 95, 96], _extractor(), gen, _cfg(tmp_path, 1), _args(), "m")
    assert drafts == {94: "draft stage1_siman_94", 95: "sync draft 95", 96: "draft stage1_siman_96"}
    assert gen.sync_calls == [("draft", 95)]          # only the failed siman, one sync retry
    assert _saved(tmp_path) == [94, 95, 96]


def test_stage1_failure_saves_the_others_then_aborts(tmp_path, caplog):
    gen = FakeGenerator(batch=lambda cid: REQUEST_ERROR if cid.endswith("_95") else f"draft {cid}",
                        sync=lambda kind, key: RuntimeError("503 overloaded"))
    with pytest.raises(SystemExit) as exc:
        main_mod.generate_drafts([94, 95, 96, 97], _extractor(), gen, _cfg(tmp_path, 1), _args(), "m")
    assert exc.value.code == 1
    assert _saved(tmp_path) == [94, 96, 97]           # a failure no longer loses the later results
    assert (tmp_path / "YD_Siman_96_m_draft.txt").read_text(encoding="utf-8") == "draft stage1_siman_96"
    assert "error code 13: Internal error" in caplog.text
    assert "Siman 95 | stage: Stage 1 (Drafts) | reason: 503 overloaded (after 2 attempt(s))" in caplog.text
    assert "3 result(s) were saved to the cache" in caplog.text


def test_stage1_sync_mode_retries_too(tmp_path):
    answers = iter([RuntimeError("deadline"), "draft"])
    gen = FakeGenerator(sync=lambda kind, key: next(answers))
    assert main_mod.generate_drafts([94], _extractor(), gen, _cfg(tmp_path, 1), _args(batch=False), "m") == {94: "draft"}
    assert gen.submitted == [] and len(gen.sync_calls) == 2


def test_stage1_context_failure_aborts_after_other_drafts(tmp_path):
    extractor = MagicMock()
    extractor.compile_simanim_context.side_effect = lambda simanim, target_simanim: (
        (_ for _ in ()).throw(RuntimeError("sefaria down")) if simanim == [95] else "ctx")
    gen = FakeGenerator()
    with pytest.raises(SystemExit):
        main_mod.generate_drafts([94, 95], extractor, gen, _cfg(tmp_path, 0), _args(), "m")
    assert gen.submitted == ["stage1_siman_94"] and _saved(tmp_path) == [94]


def test_stage2_batch_failure_retried_then_aborts(tmp_path):
    gen = FakeGenerator(batch=lambda cid: ValueError("no candidates"))
    assert main_mod.generate_relations({94: "d"}, gen, _cfg(tmp_path, 1), _args()) == "sync relations (94,)"
    gen = FakeGenerator(batch=lambda cid: ValueError("no candidates"), sync=lambda kind, key: RuntimeError("down"))
    with pytest.raises(SystemExit):
        main_mod.generate_relations({94: "d"}, gen, _cfg(tmp_path, 1), _args())


def test_stage3_polish_batch_error_retried_and_counted(tmp_path):
    gen = FakeGenerator(batch=lambda cid: REQUEST_ERROR if cid.endswith("_95") else f"polished {cid}",
                        sync=lambda kind, key: RuntimeError("still failing"))
    summary = main_mod.RunSummary()
    polished = main_mod.polish_simanim({94: "d94", 95: "d95"}, "rel", gen, _cfg(tmp_path, 2), _args(), summary)
    assert polished == {94: "polished stage3_siman_94"}
    assert gen.sync_calls == [("polish", "d95")] * 2   # max_retries = 2
    assert summary.failed[95] == (main_mod.STAGE_POLISH, "still failing (after 3 attempt(s))")
