"""Stage 3+ flow in main.py: polish -> pronunciation tagging -> transcript -> TTS, sync and batch."""
from unittest.mock import MagicMock, patch

import pytest

import main as main_mod
from pipeline.generator import BatchCapableGenerator
from pipeline.pronunciation import PronunciationTagger

TET = '<phoneme alphabet="ipa" ph="tet">טית</phoneme>'


def tagged(text):
    return text.replace("טית", TET)


class FakeBatchGenerator(BatchCapableGenerator):
    """Batch generator whose job outcome is decided by `respond(custom_id, user_prompt)`."""

    def __init__(self, respond, model_name="m", temperature=0.2, failed_ids=()):
        self.respond = respond
        self.model_name = model_name
        self.temperature = temperature
        self.failed_ids = set(failed_ids)
        self.submitted = []
        self._results = {}

    def submit_batch(self, system_instruction, user_prompt, temperature, custom_id, cache_dir="./cache",
                     thinking_level=None):
        self.submitted.append(dict(system_instruction=system_instruction, user_prompt=user_prompt,
                                   temperature=temperature, custom_id=custom_id, thinking_level=thinking_level))
        self._results[custom_id] = (custom_id, user_prompt)
        return custom_id

    def get_batch_status(self, job_id):
        return "failed" if job_id in self.failed_ids else "completed"

    def get_batch_result(self, job_id):
        return self.respond(*self._results[job_id])

    # Stage 3 polishing in batch mode builds its prompt through the generator
    def _get_polishing_user_prompt(self, draft, relations_text):
        return draft


def _config(tmp_path):
    return MagicMock(output_dir=str(tmp_path), section_slug="YD", cache_dir=str(tmp_path),
                     polishing_instruction="POLISH")


def _args(batch):
    return MagicMock(batch=batch, skip_tts=True, batch_poll_interval=0, batch_timeout=60)


DRAFTS = {94: "טית 94", 95: "טית 95", 96: "טית 96", 97: "טית 97"}


def _run(tmp_path, generator, tagger, batch, drafts=DRAFTS, targets=None):
    return main_mod.run_lesson_stages(targets or list(drafts), drafts, "rel", generator, tagger,
                                      _config(tmp_path), _args(batch), "m")


def _transcripts(tmp_path):
    return sorted(int(p.name.split("_")[2]) for p in tmp_path.glob("YD_Siman_*_m_transcript.txt"))


# --- batch -----------------------------------------------------------------------

def test_batch_tagging_success(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    tag_gen = FakeBatchGenerator(lambda cid, prompt: tagged(prompt), temperature=0.35)
    summary = _run(tmp_path, polish_gen, PronunciationTagger(tag_gen, "TAG", "low"), batch=True)
    assert summary.succeeded == [94, 95, 96, 97] and not summary.failed
    assert [r["custom_id"] for r in tag_gen.submitted] == [f"stage3_4_siman_{s}" for s in DRAFTS]
    assert all(r["system_instruction"] == "TAG" and r["temperature"] == 0.35 and r["thinking_level"] == "low"
               for r in tag_gen.submitted)
    assert TET in (tmp_path / "YD_Siman_94_m_pronunciation.txt").read_text(encoding="utf-8")
    assert _transcripts(tmp_path) == [94, 95, 96, 97]


def test_batch_tagging_failed_job_fails_that_siman(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    tag_gen = FakeBatchGenerator(lambda cid, prompt: tagged(prompt), failed_ids={"stage3_4_siman_95"})
    summary = _run(tmp_path, polish_gen, PronunciationTagger(tag_gen, "TAG", None), batch=True)
    assert summary.failed == {95: (main_mod.STAGE_PRONUNCIATION, "batch job failed (job_id: stage3_4_siman_95)")}
    assert _transcripts(tmp_path) == [94, 96, 97]
    assert not (tmp_path / "YD_Siman_95_m_pronunciation.txt").exists()


def test_batch_tagging_validation_failure_and_empty_result(tmp_path):
    def respond(cid, prompt):
        if cid.endswith("_95"):
            return prompt.replace("95", "99")  # text changed outside tags
        if cid.endswith("_96"):
            raise ValueError("Gemini batch returned empty text response.")
        return tagged(prompt)
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    summary = _run(tmp_path, polish_gen, PronunciationTagger(FakeBatchGenerator(respond), "TAG", None), batch=True)
    assert set(summary.failed) == {95, 96}
    stage, reason = summary.failed[95]
    assert stage == main_mod.STAGE_PRONUNCIATION and "text changed outside tags" in reason
    stage, reason = summary.failed[96]
    assert stage == main_mod.STAGE_PRONUNCIATION and "empty text response" in reason
    assert _transcripts(tmp_path) == [94, 97]


def test_batch_mixed_simanim_summary(tmp_path, caplog):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt, failed_ids={"stage3_siman_95"})
    tag_gen = FakeBatchGenerator(lambda cid, prompt: prompt.replace("טית", "<phoneme>") if cid.endswith("_96")
                                 else tagged(prompt))
    drafts = {94: "טית 94", 95: "טית 95", 96: "טית 96"}
    summary = _run(tmp_path, polish_gen, PronunciationTagger(tag_gen, "TAG", None), batch=True,
                   drafts=drafts, targets=[93, 94, 95, 96])
    assert summary.succeeded == [94]
    assert summary.failed[93] == (main_mod.STAGE_POLISH, "Stage 1 draft is missing")
    assert summary.failed[95] == (main_mod.STAGE_POLISH, "batch job failed (job_id: stage3_siman_95)")
    assert summary.failed[96][0] == main_mod.STAGE_PRONUNCIATION
    assert "malformed phoneme tag" in summary.failed[96][1]
    # a siman that failed polish is never sent for tagging
    assert [r["custom_id"] for r in tag_gen.submitted] == ["stage3_4_siman_94", "stage3_4_siman_96"]
    text = summary.format()
    assert "Succeeded (1/4): 94" in text and "Failed (3/4)" in text
    assert "Siman 96 | stage: Stage 3.4 (Pronunciation tagging) | reason: pronunciation tagging output rejected" in text


def test_batch_tagging_disabled_passes_text_through(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    summary = _run(tmp_path, polish_gen, None, batch=True)
    assert summary.succeeded == list(DRAFTS)
    assert not list(tmp_path.glob("*pronunciation*"))


# --- sync (unchanged behaviour) -----------------------------------------------------

def test_sync_mode_tags_each_siman_synchronously(tmp_path):
    gen = MagicMock()
    gen.polish_siman_script.side_effect = lambda draft, rel, instr: draft
    tag_gen = MagicMock(model_name="m", temperature=0.2)
    tag_gen.generate_text.side_effect = lambda instr, text, thinking_level: (
        text.replace("95", "99") if "95" in text else tagged(text))
    summary = _run(tmp_path, gen, PronunciationTagger(tag_gen, "TAG", "high"), batch=False)
    assert summary.succeeded == [94, 96, 97]
    assert summary.failed[95][0] == main_mod.STAGE_PRONUNCIATION
    assert tag_gen.generate_text.call_count == 4
    assert all(c.kwargs == {"thinking_level": "high"} for c in tag_gen.generate_text.call_args_list)
    tag_gen.submit_batch.assert_not_called()
    gen.submit_batch.assert_not_called()
    assert _transcripts(tmp_path) == [94, 96, 97]


# --- startup check ------------------------------------------------------------------

def test_batch_with_non_batch_pronunciation_engine_errors_at_startup(tmp_path, monkeypatch, caplog):
    cfg = MagicMock(section_slug="YD", drafts_dir=str(tmp_path), relations_dir=str(tmp_path),
                    output_dir=str(tmp_path), pronunciation_enabled=True, pronunciation_engine="custom")
    monkeypatch.setattr("sys.argv", ["main.py", "94", "--batch"])
    non_batch_tagger = PronunciationTagger(object(), "TAG", None)
    with patch.object(main_mod, "PipelineConfig", return_value=cfg), \
         patch.object(main_mod, "create_generator_engine", return_value=FakeBatchGenerator(None)), \
         patch.object(main_mod, "create_pronunciation_tagger", return_value=non_batch_tagger), \
         patch.object(main_mod, "SefariaExtractor") as extractor:
        with pytest.raises(SystemExit) as exc:
            main_mod.main()
    assert exc.value.code == 1
    assert "pronunciation engine that supports batch" in caplog.text
    extractor.assert_not_called()  # failed before any stage ran
