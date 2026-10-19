"""Batch results: per-request errors (Gemini / OpenAI) and the shared retry path of Stages 1, 2, 3."""
import json
from unittest.mock import MagicMock, patch, mock_open

import pytest
from google.genai import types

import main as main_mod
from pipeline.generator import BatchCapableGenerator, BatchResultError, GeminiScriptGenerator, OpenAIScriptGenerator
from pipeline.retry import call_with_retries
from tests.gemini_fakes import make_gemini_generator


# --- Gemini: InlinedResponse carries `response` OR `error` ---------------------------

def _gemini_with(*inlined):
    gen, client = make_gemini_generator("unused", model_name="gemini-pro", temperature=0.2)
    job = types.BatchJob(name="batches/1", dest=types.BatchJobDestination(inlined_responses=list(inlined)))
    client.batches.get.return_value = job
    return gen


def _response(*parts, finish=types.FinishReason.STOP, **kwargs):
    content = types.Content(role="model", parts=list(parts)) if parts else None
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content, finish_reason=finish)], **kwargs)


def test_gemini_batch_result_text_without_thought_parts():
    gen = _gemini_with(types.InlinedResponse(response=_response(
        types.Part(text="thinking...", thought=True), types.Part(text="טיוטה "), types.Part(text="מלאה"))))
    assert gen.get_batch_result("batches/1") == "טיוטה מלאה"


def test_gemini_batch_request_error_is_surfaced():
    # what the API returned for 3 of 11 Stage 1 requests inside a SUCCEEDED job
    gen = _gemini_with(types.InlinedResponse(error=types.JobError(code=1, message="The operation was cancelled.")))
    with pytest.raises(BatchResultError) as exc:
        gen.get_batch_result("batches/1")
    assert str(exc.value) == ("Gemini batch request failed (job batches/1): "
                              "error code 1 (CANCELLED): The operation was cancelled.")
    gen = _gemini_with(types.InlinedResponse(error=types.JobError(code=99, message="x", details=["d1"])))
    with pytest.raises(BatchResultError, match=r"error code 99: x \(details: d1\)$"):
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
    with pytest.raises(BatchResultError, match=reason):
        _gemini_with(inlined).get_batch_result("batches/1")


def test_gemini_batch_job_without_responses():
    with pytest.raises(BatchResultError, match="has no inlined response"):
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
    with pytest.raises(BatchResultError, match=reason):
        _openai_with(output=output, error=error).get_batch_result("b")


# --- retry helper ------------------------------------------------------------------

def test_call_with_retries(caplog):
    answers = iter([ValueError("again"), "ok"])
    def call():
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer
    assert call_with_retries(call, 2, "Siman 94 X") == "ok"
    assert "Siman 94 X attempt 1/3 failed: again; retrying" in caplog.text


def test_call_with_retries_raises_last_reason():
    with pytest.raises(RuntimeError, match=r"^last \(after 2 attempt\(s\)\)$"):
        call_with_retries(MagicMock(side_effect=[ValueError("first"), ValueError("last")]), 1, "x")


# --- config ------------------------------------------------------------------------

BASE_YAML = """
halachic_section: "Yoreh De'ah"
directories:
  thinking_dir: "./output/thinking"
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
    """
    Batch results come from batch(custom_id, n), n = how many times that custom_id was submitted;
    sync calls from sync(kind, key). A returned Exception is raised.
    """

    def __init__(self, batch=lambda cid, n: f"draft {cid}", sync=lambda kind, key: f"sync {kind} {key}",
                 failed_jobs=()):
        self.batch, self.sync, self.failed_jobs = batch, sync, set(failed_jobs)
        self.model_name, self.temperature = "m", 0.2
        self.sync_calls, self.submitted = [], []
        self.sync_thinking, self.submitted_thinking = [], []
        self.status_errors, self.fetch_errors = {}, {}  # job_id -> exceptions raised by consecutive calls

    def _answer(self, value):
        if isinstance(value, Exception):
            raise value
        return value

    def submit_batch(self, system_instruction, user_prompt, temperature, custom_id, cache_dir="./cache",
                     thinking_level=None):
        self.submitted.append(custom_id)
        self.submitted_thinking.append((custom_id, thinking_level))
        return f"{custom_id}#{self.submitted.count(custom_id)}"

    def get_batch_status(self, job_id):
        if self.status_errors.get(job_id):
            raise self.status_errors[job_id].pop(0)
        return "failed" if job_id in self.failed_jobs else "completed"

    def get_batch_result(self, job_id):
        if self.fetch_errors.get(job_id):
            raise self.fetch_errors[job_id].pop(0)
        custom_id, n = job_id.split("#")
        return self._answer(self.batch(custom_id, int(n)))

    def _sync(self, kind, key):
        self.sync_calls.append((kind, key))
        return self._answer(self.sync(kind, key))

    def generate_siman_script(self, siman, master_context, system_instruction, commentators_desc,
                              thinking_level, thinking_name):
        self.sync_thinking.append((siman, thinking_level, thinking_name))
        return self._sync("draft", siman)

    def analyze_cross_relations(self, drafts, relations_instruction, thinking_name):
        self.sync_thinking.append(("relations", None, thinking_name))
        return self._sync("relations", tuple(drafts))

    def polish_siman_script(self, script_text, relations_text, polishing_instruction, thinking_name):
        self.sync_thinking.append(("polish", None, thinking_name))
        return self._sync("polish", script_text)

    def _get_generation_user_prompt(self, gematria_siman, master_context, commentators_desc):
        return master_context

    def _get_relations_user_prompt(self, drafts):
        return "rel"

    def _get_polishing_user_prompt(self, draft, relations_text):
        return draft


CANCELLED = "Gemini batch request failed (job {}): error code 1 (CANCELLED): The operation was cancelled."


def cancelled(cid, rounds=(1,)):
    """batch() answer: the request is CANCELLED in the given submission rounds, else a draft."""
    return lambda c, n: BatchResultError(CANCELLED.format(c)) if c == cid and n in rounds else f"draft {c} #{n}"


def _cfg(tmp_path, retries):
    return MagicMock(drafts_dir=str(tmp_path), cache_dir=str(tmp_path), section_slug="YD",
                     generator_max_retries=retries, section_metadata={"prompt_commentators_desc": "d"},
                     draft_thinking_level="medium")


def _args(batch=True, timeout=60):
    return MagicMock(batch=batch, batch_poll_interval=0, batch_timeout=timeout)


def _saved(tmp_path):
    return sorted(int(p.name.split("_")[2]) for p in tmp_path.glob("YD_Siman_*_m_draft.txt"))


def _extractor():
    return MagicMock(compile_simanim_context=lambda simanim, target_simanim: f"context {simanim[0]}")


def test_stage1_failed_requests_resubmitted_in_one_batch_round(tmp_path, caplog):
    def batch(cid, n):
        if cid in ("stage1_siman_95", "stage1_siman_97") and n == 1:
            return BatchResultError(CANCELLED.format(cid))
        return f"draft {cid} #{n}"
    gen = FakeGenerator(batch=batch)
    drafts = main_mod.generate_drafts([94, 95, 96, 97], _extractor(), gen, _cfg(tmp_path, 1), _args(), "m")
    assert drafts == {94: "draft stage1_siman_94 #1", 95: "draft stage1_siman_95 #2",
                      96: "draft stage1_siman_96 #1", 97: "draft stage1_siman_97 #2"}
    assert list(drafts) == [94, 95, 96, 97]           # siman order kept
    # round 2 resubmits both failed simanim together; batch mode never calls sync
    assert gen.submitted == ["stage1_siman_94", "stage1_siman_95", "stage1_siman_96", "stage1_siman_97",
                             "stage1_siman_95", "stage1_siman_97"]
    assert gen.sync_calls == []
    assert _saved(tmp_path) == [94, 95, 96, 97]
    assert ("Siman 95 Stage 1 (Drafts) attempt 1/2 failed: Gemini batch request failed (job stage1_siman_95): "
            "error code 1 (CANCELLED): The operation was cancelled.; retrying.") in caplog.text
    assert "Stage 1 (Drafts): retry round 1/1 for 2 item(s) as new batch jobs." in caplog.text


def test_stage1_failure_saves_the_others_then_aborts(tmp_path, caplog):
    gen = FakeGenerator(batch=cancelled("stage1_siman_95", rounds=(1, 2)))
    with pytest.raises(SystemExit) as exc:
        main_mod.generate_drafts([94, 95, 96, 97], _extractor(), gen, _cfg(tmp_path, 1), _args(), "m")
    assert exc.value.code == 1
    assert _saved(tmp_path) == [94, 96, 97]           # a failure no longer loses the later results
    assert (tmp_path / "YD_Siman_96_m_draft.txt").read_text(encoding="utf-8") == "draft stage1_siman_96 #1"
    assert gen.submitted.count("stage1_siman_95") == 2 and gen.sync_calls == []
    assert ("Siman 95 | stage: Stage 1 (Drafts) | reason: Gemini batch request failed (job stage1_siman_95): "
            "error code 1 (CANCELLED): The operation was cancelled. (after 2 attempt(s))") in caplog.text
    assert "3 draft(s) were saved to the cache; a rerun only regenerates the missing ones." in caplog.text


def test_stage1_no_retry_round_with_zero_budget(tmp_path):
    gen = FakeGenerator(batch=cancelled("stage1_siman_95"))
    with pytest.raises(SystemExit):
        main_mod.generate_drafts([94, 95], _extractor(), gen, _cfg(tmp_path, 0), _args(), "m")
    assert gen.submitted == ["stage1_siman_94", "stage1_siman_95"] and _saved(tmp_path) == [94]


def test_stage1_failed_job_and_submit_error_are_resubmitted(tmp_path):
    gen = FakeGenerator(failed_jobs={"stage1_siman_94#1"})
    submit = gen.submit_batch
    calls = []
    def flaky_submit(**kwargs):
        calls.append(kwargs["custom_id"])
        if calls.count("stage1_siman_95") == 1 and kwargs["custom_id"] == "stage1_siman_95":
            raise RuntimeError("429 quota")
        return submit(**kwargs)
    gen.submit_batch = flaky_submit
    drafts = main_mod.generate_drafts([94, 95], _extractor(), gen, _cfg(tmp_path, 1), _args(), "m")
    assert drafts == {94: "draft stage1_siman_94", 95: "draft stage1_siman_95"}
    assert calls == ["stage1_siman_94", "stage1_siman_95", "stage1_siman_94", "stage1_siman_95"]


def test_stage1_sync_mode_retries_with_sync_calls(tmp_path):
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


def test_stage1_save_error_fails_siman_without_resubmission(tmp_path):
    gen = FakeGenerator()
    with patch.object(main_mod, "save_draft", side_effect=OSError("disk full")), pytest.raises(SystemExit):
        main_mod.generate_drafts([94], _extractor(), gen, _cfg(tmp_path, 2), _args(), "m")
    assert gen.submitted == ["stage1_siman_94"]


def test_stage2_batch_failure_resubmitted_then_aborts(tmp_path):
    gen = FakeGenerator(batch=lambda cid, n: BatchResultError("no candidates") if n == 1 else "relations")
    assert main_mod.generate_relations({94: "d"}, gen, _cfg(tmp_path, 1), _args(), "YD_Relations_94_m") == "relations"
    assert gen.submitted == ["stage2_relations"] * 2 and gen.sync_calls == []
    gen = FakeGenerator(batch=lambda cid, n: BatchResultError("no candidates"))
    with pytest.raises(SystemExit):
        main_mod.generate_relations({94: "d"}, gen, _cfg(tmp_path, 1), _args(), "YD_Relations_94_m")


def test_stage3_polish_batch_retry_budget_then_fails_that_siman(tmp_path):
    gen = FakeGenerator(batch=cancelled("stage3_siman_95", rounds=(1, 2, 3)))
    summary = main_mod.RunSummary()
    polished = main_mod.polish_simanim({94: "d94", 95: "d95"}, "rel", gen, _cfg(tmp_path, 2), _args(), summary, "m")
    assert polished == {94: "draft stage3_siman_94 #1"}
    assert gen.submitted == ["stage3_siman_94"] + ["stage3_siman_95"] * 3   # max_retries = 2 rounds
    assert gen.sync_calls == []
    assert summary.failed[95] == (main_mod.STAGE_POLISH, CANCELLED.format("stage3_siman_95") + " (after 3 attempt(s))")


def test_stage1_sends_draft_thinking_level_in_sync_and_batch(tmp_path):
    gen = FakeGenerator(batch=lambda c, n: f"draft {c}")
    main_mod.generate_drafts([94], _extractor(), gen, _cfg(tmp_path, 0), _args(), "m")
    assert gen.submitted_thinking == [("stage1_siman_94", "medium")]
    gen = FakeGenerator(sync=lambda kind, key: f"draft {key}")
    main_mod.generate_drafts([94], _extractor(), gen, _cfg(tmp_path, 0), _args(batch=False), "m")
    assert gen.sync_thinking == [(94, "medium", "YD_Siman_94_m_draft")]


def test_stage2_and_3_sync_calls_name_their_thinking_files(tmp_path):
    gen = FakeGenerator(sync=lambda kind, key: "out")
    main_mod.generate_relations({94: "d"}, gen, _cfg(tmp_path, 0), _args(batch=False), "YD_Relations_94_m")
    main_mod.polish_simanim({94: "d94"}, "rel", gen, _cfg(tmp_path, 0), _args(batch=False), main_mod.RunSummary(), "m")
    assert gen.sync_thinking == [("relations", None, "YD_Relations_94_m"), ("polish", None, "YD_Siman_94_m_polish")]


# --- a failed status check / result fetch says nothing about the job: never resubmitted -------

def test_status_check_error_keeps_polling_without_resubmitting(tmp_path, caplog):
    gen = FakeGenerator()
    gen.status_errors["stage1_siman_95#1"] = [ConnectionError("Connection reset by peer")] * 2
    drafts = main_mod.generate_drafts([94, 95], _extractor(), gen, _cfg(tmp_path, 1), _args(), "m")
    assert drafts == {94: "draft stage1_siman_94", 95: "draft stage1_siman_95"}
    assert gen.submitted == ["stage1_siman_94", "stage1_siman_95"]
    assert ("Stage 1 (Drafts) batch job for Siman 95 (job_id: stage1_siman_95#1): status check failed: "
            "Connection reset by peer; status unknown, checking again at the next poll.") in caplog.text
    assert "attempt 1/2 failed" not in caplog.text


def test_result_fetch_error_is_fetched_again_not_resubmitted(tmp_path, caplog):
    gen = FakeGenerator()
    gen.fetch_errors["stage1_siman_94#1"] = [ConnectionError("read timeout")]
    drafts = main_mod.generate_drafts([94], _extractor(), gen, _cfg(tmp_path, 1), _args(), "m")
    assert drafts == {94: "draft stage1_siman_94"} and gen.submitted == ["stage1_siman_94"]
    assert "completed, but fetching its result failed: read timeout; fetching again at the next poll." in caplog.text


def test_batch_timeout_lists_unresolved_jobs_and_keeps_saved_drafts(tmp_path):
    gen = FakeGenerator()
    gen.status_errors["stage1_siman_95#1"] = [ConnectionError("DNS failure")] * 5
    gen.fetch_errors["stage1_siman_96#1"] = [ConnectionError("read timeout")] * 5
    with pytest.raises(main_mod.BatchTimeoutError) as exc:
        main_mod.generate_drafts([94, 95, 96], _extractor(), gen, _cfg(tmp_path, 3), _args(timeout=0), "m")
    assert str(exc.value) == (
        "Stage 1 (Drafts): --batch-timeout (0s) expired with 2 unresolved batch job(s); fetch them later by "
        "job id: Siman 95: job_id stage1_siman_95#1 (status check failed: DNS failure); "
        "Siman 96: job_id stage1_siman_96#1 (completed, result fetch failed: read timeout)")
    assert gen.submitted == ["stage1_siman_94", "stage1_siman_95", "stage1_siman_96"]   # nothing resubmitted
    assert _saved(tmp_path) == [94]                                                     # fetched draft kept


def test_batch_timeout_with_job_still_running(tmp_path):
    gen = FakeGenerator()
    gen.get_batch_status = lambda job_id: "pending"
    summary = main_mod.RunSummary()
    with pytest.raises(main_mod.BatchTimeoutError, match=r"Siman 94: job_id stage3_siman_94#1 \(still pending\)$"):
        main_mod.polish_simanim({94: "d"}, "rel", gen, _cfg(tmp_path, 1), _args(timeout=0), summary, "m")


@pytest.mark.parametrize("state, status", [
    ("JOB_STATE_SUCCEEDED", "completed"), ("JOB_STATE_PARTIALLY_SUCCEEDED", "completed"),
    ("JOB_STATE_FAILED", "failed"), ("JOB_STATE_CANCELLED", "failed"), ("JOB_STATE_EXPIRED", "failed"),
    ("JOB_STATE_CANCELLING", "pending"), ("JOB_STATE_RUNNING", "pending"), ("JOB_STATE_QUEUED", "pending"),
])
def test_gemini_batch_status(state, status):
    gen = _gemini_with()
    gen.client.batches.get.return_value = types.BatchJob(name="batches/1", state=state)
    assert gen.get_batch_status("batches/1") == status
