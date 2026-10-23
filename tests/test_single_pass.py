"""main.py --single-pass: raw Sefaria sources -> Stage 3 polish (no drafts, no relations) -> nikkud -> transcript."""
from unittest.mock import MagicMock, patch

import pytest

import main as main_mod
from pipeline.generator import GeminiScriptGenerator
from tests.test_dry_run import _args, _cfg, cache  # noqa: F401  (fixture)
from tests.test_gemini_streaming import BASE, _config

PROMPT = "מקורות {gematria_siman}:\n{sources}"


def test_plan_needs_no_drafts_or_relations(cache):
    plan = main_mod.resolve_run_plan(_args(single_pass=True), _cfg(cache), [94, 95], [94, 95], "m_single_pass")
    assert plan.relations_source == "none" and plan.relations_text == ""
    assert plan.drafts_needed == [] and plan.missing_drafts == []
    assert plan.polish_paths[94] == str(cache / "out" / "YD_Siman_94_m_single_pass_polish.txt")


def test_plan_loads_single_pass_polish_cache_only(cache):
    (cache / "out").mkdir()
    (cache / "out" / "YD_Siman_94_m_polish.txt").write_text("normal", encoding="utf-8")
    (cache / "out" / "YD_Siman_95_m_single_pass_polish.txt").write_text("single", encoding="utf-8")
    plan = main_mod.resolve_run_plan(_args(single_pass=True), _cfg(cache), [94, 95], [94, 95], "m_single_pass")
    assert plan.polished == {95: "single"} and plan.missing_polish == [94]


def test_dry_run_single_pass(cache, monkeypatch, caplog):
    from tests.test_dry_run import _main_dry_run
    _main_dry_run(cache, monkeypatch, ["94", "--single-pass"])
    out = caplog.text
    assert "--single-pass: no Stage 1 drafts and no Stage 2 relations map" in out
    assert "Stage 1 drafts needed" not in out and "Stage 2 relations map:" not in out
    assert str(cache / "out" / "YD_Siman_94_m_single_pass_polish.txt") in out
    assert str(cache / "out" / "YD_Siman_94_m_single_pass_transcript.txt") in out


@pytest.mark.parametrize("extra, flag", [
    (["--stage-1-only"], "--stage-1-only"),
    (["--relations-file", "r.txt"], "--relations-file"),
    (["--draft-file", "94=d.txt"], "--draft-file"),
    (["--context-range", "94-96"], "--context-range"),
])
def test_single_pass_conflicts_are_rejected(monkeypatch, capsys, extra, flag):
    monkeypatch.setattr("sys.argv", ["main.py", "94", "--single-pass", *extra])
    with pytest.raises(SystemExit) as exc:
        main_mod.parse_arguments()
    assert exc.value.code == 2
    assert f"--single-pass has no Stage 1 drafts or Stage 2 relations map, so it cannot be combined with {flag}" \
        in capsys.readouterr().err


def _sp_config(tmp_path):
    cfg = MagicMock(output_dir=str(tmp_path), generator_max_retries=0, section_slug="YD", cache_dir=str(tmp_path),
                    polishing_instruction="POLISH", polish_temperature=0.15)
    cfg.format_single_pass_user_prompt.side_effect = lambda g, sources: PROMPT.format(gematria_siman=g,
                                                                                      sources=sources)
    return cfg


def test_single_pass_polishes_raw_sources_sync(tmp_path):
    extractor = MagicMock()
    extractor.compile_simanim_context.side_effect = lambda simanim, target_simanim: f"sources {simanim[0]}"
    gen = MagicMock()
    gen.polish_with_prompt.side_effect = lambda prompt, instr, temperature, thinking_name: f"lesson: {prompt}"
    summary = main_mod.run_single_pass_stages([94, 95], {95: "cached lesson"}, extractor, gen, _sp_config(tmp_path),
                                              MagicMock(batch=False, skip_tts=True), "m_single_pass")
    assert summary.succeeded == [94, 95]
    extractor.compile_simanim_context.assert_called_once_with([94], target_simanim=[94])
    call = gen.polish_with_prompt.call_args
    assert call.args[:2] == ("מקורות צ\"ד:\nsources 94", "POLISH") and call.kwargs["temperature"] == 0.15
    gen.polish_siman_script.assert_not_called()
    assert (tmp_path / "YD_Siman_94_m_single_pass_polish.txt").read_text(encoding="utf-8") == \
        "lesson: מקורות צ\"ד:\nsources 94"
    assert (tmp_path / "YD_Siman_95_m_single_pass_transcript.txt").read_text(encoding="utf-8") == "cached lesson"


def test_single_pass_batch_uses_single_pass_prompt(tmp_path):
    from tests.test_stage3_flow import FakeBatchGenerator
    extractor = MagicMock()
    extractor.compile_simanim_context.return_value = "src"
    gen = FakeBatchGenerator(lambda cid, prompt: "lesson")
    args = MagicMock(batch=True, skip_tts=True, batch_poll_interval=0, batch_timeout=60)
    summary = main_mod.run_single_pass_stages([94], {}, extractor, gen, _sp_config(tmp_path), args, "m_single_pass")
    assert summary.succeeded == [94]
    assert gen.submitted == [dict(system_instruction="POLISH", user_prompt="מקורות צ\"ד:\nsrc", temperature=0.15,
                                  custom_id="stage3_single_pass_siman_94", thinking_level=None)]


def test_single_pass_sefaria_failure_skips_siman(tmp_path):
    extractor = MagicMock()
    extractor.compile_simanim_context.side_effect = RuntimeError("Sefaria down")
    gen = MagicMock()
    summary = main_mod.run_single_pass_stages([94], {}, extractor, gen, _sp_config(tmp_path),
                                              MagicMock(batch=False, skip_tts=True), "m_single_pass")
    assert summary.failed[94][0] == main_mod.STAGE_POLISH
    gen.polish_with_prompt.assert_not_called()


def test_config_single_pass_user_prompt():
    cfg = _config(BASE + 'single_pass_user_prompt: "סימן {gematria_siman}: {sources}"\n')
    assert cfg.format_single_pass_user_prompt("צד", "S {x}") == "סימן צד: S {x}"
    with pytest.raises(ValueError, match="'single_pass_user_prompt'"):
        _config(BASE).single_pass_user_prompt
    with pytest.raises(ValueError, match=r"must contain \{sources\}"):
        _config(BASE + 'single_pass_user_prompt: "סימן {gematria_siman}"\n').single_pass_user_prompt


def test_repo_config_single_pass_user_prompt():
    from pipeline.config import PipelineConfig
    prompt = PipelineConfig("config.yaml").format_single_pass_user_prompt("צד", "SRC")
    assert "סימן צד" in prompt and prompt.rstrip().endswith("SRC")


def test_polish_siman_script_wraps_polish_with_prompt():
    gen = GeminiScriptGenerator.__new__(GeminiScriptGenerator)
    with patch.object(GeminiScriptGenerator, "polish_with_prompt", return_value="out") as call:
        assert gen.polish_siman_script("d", "", "SYS", temperature=0.1, thinking_name="N") == "out"
    call.assert_called_once_with(gen._get_polishing_user_prompt("d", ""), "SYS", 0.1, "N")
