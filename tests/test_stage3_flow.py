"""Stage 3+ flow in main.py: polish and nikkud (one call) -> dictionary nikkud -> transcript (+ review) -> TTS."""
from unittest.mock import MagicMock, patch

import pytest

import main as main_mod
from pipeline.generator import BatchCapableGenerator, BatchResultError
from pipeline.nikkud_review import format_review
from pipeline.apply_nikkud import apply_nikkud_with_overrides

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
    return MagicMock(output_dir=str(tmp_path), generator_max_retries=0, section_slug="YD", cache_dir=str(tmp_path),
                     polishing_instruction="POLISH", polish_temperature=0.15)


def _args(batch):
    return MagicMock(batch=batch, skip_tts=True, batch_poll_interval=0, batch_timeout=60)


DRAFTS = {94: "טית 94", 95: "טית 95", 96: "טית 96", 97: "טית 97"}


def _run(tmp_path, generator, batch, drafts=DRAFTS, targets=None, cached_polish=None):
    return main_mod.run_lesson_stages(targets or list(drafts), drafts, cached_polish or {}, "rel", generator,
                                      _config(tmp_path), _args(batch), "m")


def _transcripts(tmp_path):
    return sorted(int(p.name.split("_")[2]) for p in tmp_path.glob("YD_Siman_*_m_transcript.txt"))


def _review(tmp_path, siman):
    return (tmp_path / f"YD_Siman_{siman}_m_pronunciation_changes.txt").read_text(encoding="utf-8")


# --- batch -----------------------------------------------------------------------

def test_batch_combined_polish_success(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: vocalized(prompt))  # the polish vocalizes
    summary = _run(tmp_path, polish_gen, batch=True)
    assert summary.succeeded == [94, 95, 96, 97] and not summary.failed
    # one LLM call per siman: the polish, with its own instruction and temperature
    assert [r["custom_id"] for r in polish_gen.submitted] == [f"stage3_siman_{s}" for s in DRAFTS]
    assert all(r["system_instruction"] == "POLISH" and r["temperature"] == 0.15 and r["thinking_level"] is None
               for r in polish_gen.submitted)
    assert (tmp_path / "YD_Siman_94_m_polish.txt").read_text(encoding="utf-8") == f"{TET} 94"  # vocalized polish
    assert (tmp_path / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8") == f"{TET} 94"
    assert _review(tmp_path, 94).splitlines()[1] == f"no\tטית\t{TET}\t1\t1\t[{TET}] 94"
    assert _transcripts(tmp_path) == [94, 95, 96, 97]


def test_batch_failed_polish_job_fails_that_siman(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: vocalized(prompt), failed_ids={"stage3_siman_95"})
    summary = _run(tmp_path, polish_gen, batch=True)
    assert summary.failed == {95: (main_mod.STAGE_POLISH, "batch job failed (job_id: stage3_siman_95) (after 1 attempt(s))")}
    assert _transcripts(tmp_path) == [94, 96, 97]
    assert not (tmp_path / "YD_Siman_95_m_pronunciation_changes.txt").exists()


def test_batch_mixed_simanim_summary(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt, failed_ids={"stage3_siman_95"})
    drafts = {94: "טית 94", 95: "טית 95", 96: "טית 96"}
    real = apply_nikkud_with_overrides
    def dictionary(text):
        if "96" in text:
            raise ValueError("bad text")
        return real(text)
    with patch.object(main_mod, "apply_nikkud_with_overrides", side_effect=dictionary):
        summary = _run(tmp_path, polish_gen, batch=True, drafts=drafts, targets=[93, 94, 95, 96])
    assert summary.succeeded == [94]
    assert summary.failed[93] == (main_mod.STAGE_POLISH, "Stage 1 draft is missing")
    assert summary.failed[95] == (main_mod.STAGE_POLISH, "batch job failed (job_id: stage3_siman_95) (after 1 attempt(s))")
    assert summary.failed[96] == (main_mod.STAGE_NIKKUD, "bad text")
    text = summary.format()
    assert "Succeeded (1/4): 94" in text and "Failed (3/4)" in text
    assert "Siman 96 | stage: Stage 3.1 (Dictionary nikkud) | reason: bad text" in text


# --- sync ------------------------------------------------------------------------

def test_sync_mode_polishes_each_siman_synchronously(tmp_path):
    gen = MagicMock()
    gen.polish_siman_script.side_effect = lambda draft, rel, instr, temperature, thinking_name: (
        _raise(RuntimeError("LLM down")) if "95" in draft else vocalized(draft))
    summary = _run(tmp_path, gen, batch=False)
    assert summary.succeeded == [94, 96, 97]
    assert summary.failed[95] == (main_mod.STAGE_POLISH, "LLM down (after 1 attempt(s))")
    assert gen.polish_siman_script.call_count == 4
    assert all(c.args[2] == "POLISH" and c.kwargs["temperature"] == 0.15 for c in gen.polish_siman_script.call_args_list)
    gen.submit_batch.assert_not_called()
    assert _transcripts(tmp_path) == [94, 96, 97]


def _raise(error):
    raise error


# --- dictionary nikkud over the polish's nikkud ------------------------------------

def test_dictionary_overrides_the_polish_nikkud_and_is_reported(tmp_path):
    gen = MagicMock()
    gen.polish_siman_script.side_effect = lambda draft, rel, instr, temperature, thinking_name: (
        draft + ' סימן הֵא והַשַׁ"ךְ אוסר בכֹּחַל ופַּת')
    summary = _run(tmp_path, gen, batch=False, drafts={94: "טית 94"})
    assert summary.succeeded == [94]
    polish = (tmp_path / "YD_Siman_94_m_polish.txt").read_text(encoding="utf-8")
    assert polish == 'טית 94 סימן הֵא והַשַׁ"ךְ אוסר בכֹּחַל ופַּת'  # the cache keeps the model's text
    final, overrides = apply_nikkud_with_overrides(polish)
    transcript = (tmp_path / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8")
    assert transcript == final and "הֵי" in transcript and "ופַּת" in transcript  # non-dictionary nikkud kept
    assert [o.model for o in overrides] == ["הֵא", 'והַשַׁ"ךְ', "בכֹּחַל"]
    assert _review(tmp_path, 94) == format_review(final, overrides)
