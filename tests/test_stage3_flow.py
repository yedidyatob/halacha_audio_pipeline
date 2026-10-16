"""Stage 3+ flow in main.py: polish -> dictionary nikkud -> pronunciation nikkud -> transcript -> TTS."""
from unittest.mock import MagicMock, patch

import pytest

import main as main_mod
from pipeline.generator import BatchCapableGenerator
from pipeline.pronunciation import PronunciationVocalizer

TET = "טֵית"


def vocalized(text):
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


def _run(tmp_path, generator, vocalizer, batch, drafts=DRAFTS, targets=None):
    return main_mod.run_lesson_stages(targets or list(drafts), drafts, "rel", generator, vocalizer,
                                      _config(tmp_path), _args(batch), "m")


def _transcripts(tmp_path):
    return sorted(int(p.name.split("_")[2]) for p in tmp_path.glob("YD_Siman_*_m_transcript.txt"))


# --- batch -----------------------------------------------------------------------

def test_batch_pronunciation_success(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    voc_gen = FakeBatchGenerator(lambda cid, prompt: vocalized(prompt), temperature=0.35)
    summary = _run(tmp_path, polish_gen, PronunciationVocalizer(voc_gen, "VOC", "low"), batch=True)
    assert summary.succeeded == [94, 95, 96, 97] and not summary.failed
    assert [r["custom_id"] for r in voc_gen.submitted] == [f"stage3_5_siman_{s}" for s in DRAFTS]
    assert all(r["system_instruction"] == "VOC" and r["temperature"] == 0.35 and r["thinking_level"] == "low"
               for r in voc_gen.submitted)
    assert TET in (tmp_path / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8")
    assert (tmp_path / "YD_Siman_94_m_pronunciation_changes.txt").read_text(encoding="utf-8") == f"טית\t{TET}\n"
    assert _transcripts(tmp_path) == [94, 95, 96, 97]


def test_batch_pronunciation_failed_job_fails_that_siman(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    voc_gen = FakeBatchGenerator(lambda cid, prompt: vocalized(prompt), failed_ids={"stage3_5_siman_95"})
    summary = _run(tmp_path, polish_gen, PronunciationVocalizer(voc_gen, "VOC", None), batch=True)
    assert summary.failed == {95: (main_mod.STAGE_PRONUNCIATION, "batch job failed (job_id: stage3_5_siman_95)")}
    assert _transcripts(tmp_path) == [94, 96, 97]
    assert not (tmp_path / "YD_Siman_95_m_pronunciation_changes.txt").exists()


def test_batch_pronunciation_validation_failure_and_empty_result(tmp_path):
    def respond(cid, prompt):
        if cid.endswith("_95"):
            return prompt.replace("95", "99")  # text changed
        if cid.endswith("_96"):
            raise ValueError("Gemini batch returned empty text response.")
        return vocalized(prompt)
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    summary = _run(tmp_path, polish_gen, PronunciationVocalizer(FakeBatchGenerator(respond), "VOC", None), batch=True)
    assert set(summary.failed) == {95, 96}
    stage, reason = summary.failed[95]
    assert stage == main_mod.STAGE_PRONUNCIATION and "text changed" in reason
    stage, reason = summary.failed[96]
    assert stage == main_mod.STAGE_PRONUNCIATION and "empty text response" in reason
    assert _transcripts(tmp_path) == [94, 97]


def test_batch_mixed_simanim_summary(tmp_path, caplog):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt, failed_ids={"stage3_siman_95"})
    voc_gen = FakeBatchGenerator(lambda cid, prompt: prompt.replace("טית", "<b>טית</b>") if cid.endswith("_96")
                                 else vocalized(prompt))
    drafts = {94: "טית 94", 95: "טית 95", 96: "טית 96"}
    summary = _run(tmp_path, polish_gen, PronunciationVocalizer(voc_gen, "VOC", None), batch=True,
                   drafts=drafts, targets=[93, 94, 95, 96])
    assert summary.succeeded == [94]
    assert summary.failed[93] == (main_mod.STAGE_POLISH, "Stage 1 draft is missing")
    assert summary.failed[95] == (main_mod.STAGE_POLISH, "batch job failed (job_id: stage3_siman_95)")
    assert summary.failed[96][0] == main_mod.STAGE_PRONUNCIATION
    assert "markup in output" in summary.failed[96][1]
    # a siman that failed polish is never sent to the pronunciation stage
    assert [r["custom_id"] for r in voc_gen.submitted] == ["stage3_5_siman_94", "stage3_5_siman_96"]
    text = summary.format()
    assert "Succeeded (1/4): 94" in text and "Failed (3/4)" in text
    assert "Siman 96 | stage: Stage 3.5 (Pronunciation nikkud) | reason: pronunciation output rejected" in text


def test_batch_pronunciation_disabled_passes_text_through(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    summary = _run(tmp_path, polish_gen, None, batch=True)
    assert summary.succeeded == list(DRAFTS)
    assert not list(tmp_path.glob("*pronunciation*"))


# --- sync (unchanged behaviour) -----------------------------------------------------

def test_sync_mode_vocalizes_each_siman_synchronously(tmp_path):
    gen = MagicMock()
    gen.polish_siman_script.side_effect = lambda draft, rel, instr: draft
    voc_gen = MagicMock(model_name="m", temperature=0.2)
    voc_gen.generate_text.side_effect = lambda instr, text, thinking_level: (
        text.replace("95", "99") if "95" in text else vocalized(text))
    summary = _run(tmp_path, gen, PronunciationVocalizer(voc_gen, "VOC", "high"), batch=False)
    assert summary.succeeded == [94, 96, 97]
    assert summary.failed[95][0] == main_mod.STAGE_PRONUNCIATION
    assert voc_gen.generate_text.call_count == 4
    assert all(c.kwargs == {"thinking_level": "high"} for c in voc_gen.generate_text.call_args_list)
    voc_gen.submit_batch.assert_not_called()
    gen.submit_batch.assert_not_called()
    assert _transcripts(tmp_path) == [94, 96, 97]


# --- startup check ------------------------------------------------------------------

def test_batch_with_non_batch_pronunciation_engine_errors_at_startup(tmp_path, monkeypatch, caplog):
    cfg = MagicMock(section_slug="YD", drafts_dir=str(tmp_path), relations_dir=str(tmp_path),
                    output_dir=str(tmp_path), pronunciation_enabled=True, pronunciation_engine="custom")
    monkeypatch.setattr("sys.argv", ["main.py", "94", "--batch"])
    non_batch_vocalizer = PronunciationVocalizer(object(), "VOC", None)
    with patch.object(main_mod, "PipelineConfig", return_value=cfg), \
         patch.object(main_mod, "create_generator_engine", return_value=FakeBatchGenerator(None)), \
         patch.object(main_mod, "create_pronunciation_vocalizer", return_value=non_batch_vocalizer), \
         patch.object(main_mod, "SefariaExtractor") as extractor:
        with pytest.raises(SystemExit) as exc:
            main_mod.main()
    assert exc.value.code == 1
    assert "pronunciation engine that supports batch" in caplog.text
    extractor.assert_not_called()  # failed before any stage ran


def test_dictionary_nikkud_runs_before_llm_stage(tmp_path):
    gen = MagicMock()
    gen.polish_siman_script.side_effect = lambda draft, rel, instr: draft + ' סימן הא והש"ך'
    voc_gen = MagicMock(model_name="m", temperature=0.2)
    voc_gen.generate_text.side_effect = lambda instr, text, thinking_level: text
    summary = _run(tmp_path, gen, PronunciationVocalizer(voc_gen, "VOC", None), batch=False, drafts={94: "טית 94"})
    assert summary.succeeded == [94]
    sent = voc_gen.generate_text.call_args.args[1]
    assert sent == "טית 94 סימן הֵי והשַׁךְ"  # respelling + dictionary nikkud already applied
    assert (tmp_path / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8") == sent
