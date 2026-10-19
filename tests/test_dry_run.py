"""main.py --dry-run: the run plan (resolve_run_plan, shared with a real run) without any API call."""
from unittest.mock import MagicMock, patch

import pytest

import main as main_mod
from pipeline.generator import GeminiScriptGenerator


def _cfg(tmp_path):
    return MagicMock(section_slug="YD", drafts_dir=str(tmp_path / "drafts"), relations_dir=str(tmp_path / "rel"),
                     output_dir=str(tmp_path / "out"), generator_engine="gemini", tts_engine="gemini",
                     pronunciation_enabled=False)


def _args(**kw):
    defaults = dict(relations_file=None, overwrite_cache=False, limit_lessons=None, batch=False,
                    stage_1_only=False, skip_tts=False)
    return MagicMock(**{**defaults, **kw})


@pytest.fixture
def cache(tmp_path):
    (tmp_path / "drafts").mkdir()
    (tmp_path / "rel").mkdir()
    for s in (94, 96):
        (tmp_path / "drafts" / f"YD_Siman_{s}_m_draft.txt").write_text(f"draft {s}", encoding="utf-8")
    return tmp_path


def _rel(tmp_path, context=(94, 95, 96)):
    return tmp_path / "rel" / f"YD_Relations_{'_'.join(map(str, context))}_m.txt"


def test_plan_without_relations_needs_whole_context(cache):
    plan = main_mod.resolve_run_plan(_args(), _cfg(cache), [94], [94, 95, 96], "m")
    assert plan.relations_source == "generate" and plan.relations_path == str(_rel(cache))
    assert plan.drafts_needed == [94, 95, 96]
    assert plan.drafts == {94: "draft 94", 96: "draft 96"} and plan.missing_drafts == [95]


def test_plan_with_cached_relations_needs_only_targets(cache):
    _rel(cache).write_text("rel", encoding="utf-8")
    plan = main_mod.resolve_run_plan(_args(), _cfg(cache), [94, 95], [94, 95, 96], "m")
    assert plan.relations_source == "cache" and plan.relations_text == "rel"
    assert plan.drafts_needed == [94, 95] and plan.missing_drafts == [95]


def test_plan_overwrite_cache_regenerates_everything(cache):
    _rel(cache).write_text("rel", encoding="utf-8")
    plan = main_mod.resolve_run_plan(_args(overwrite_cache=True), _cfg(cache), [94], [94, 95, 96], "m")
    assert plan.relations_source == "generate" and plan.drafts == {} and plan.missing_drafts == [94, 95, 96]


def test_plan_relations_file_and_limit_lessons(cache):
    rel_file = cache / "my_rel.txt"
    rel_file.write_text("mine", encoding="utf-8")
    plan = main_mod.resolve_run_plan(_args(relations_file=str(rel_file), limit_lessons=1), _cfg(cache),
                                     [95, 96], [94, 95, 96], "m")
    assert plan.relations_source == "file" and plan.relations_text == "mine"
    assert plan.target_simanim == [95] and plan.drafts_needed == [95] and plan.missing_drafts == [95]


@pytest.mark.parametrize("kw, message", [
    (dict(relations_file="nope.txt"), "Relations file not found"),
    (dict(limit_lessons=0), "greater than zero"),
])
def test_plan_errors(cache, kw, message):
    with pytest.raises(main_mod.PlanError, match=message):
        main_mod.resolve_run_plan(_args(**kw), _cfg(cache), [94], [94], "m")


def _main_dry_run(cache, monkeypatch, argv):
    gen = MagicMock(spec=GeminiScriptGenerator, model_name="m")  # batch-capable
    monkeypatch.setattr("sys.argv", ["main.py", *argv, "--dry-run"])
    with patch.object(main_mod, "PipelineConfig", return_value=_cfg(cache)), \
         patch.object(main_mod, "create_generator_engine", return_value=gen), \
         patch.object(main_mod, "SefariaExtractor") as extractor, \
         patch.object(main_mod, "create_tts_engine") as tts, \
         patch.object(main_mod, "save_output_file") as save, \
         pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 0
    # nothing beyond the plan: no Sefaria, LLM, TTS call and no file written
    extractor.assert_not_called()
    tts.assert_not_called()
    save.assert_not_called()
    assert gen.method_calls == []
    return gen


def test_main_dry_run_prints_plan_and_calls_nothing(cache, monkeypatch, caplog):
    _main_dry_run(cache, monkeypatch, ["94", "--context-range", "94-96", "--batch"])
    out = caplog.text
    assert "Dry run: plan" in out
    assert f"Stage 2 relations map: will be GENERATED (a batch job) from the drafts of [94, 95, 96] " \
           f"and saved to {_rel(cache)}" in out
    assert "from cache (2): [94, 96]" in out
    assert "to be GENERATED (1, as batch jobs, after a Sefaria fetch each): [95]" in out
    assert f"Siman 95 -> {cache / 'drafts' / 'YD_Siman_95_m_draft.txt'}" in out
    assert "3.5 pronunciation nikkud: [94]" in out and "disabled" in out
    assert f"Siman 94: {cache / 'out' / 'YD_Siman_94_m'}_transcript.txt, {cache / 'out' / 'YD_Siman_94_m'}.mp3" in out


def test_main_dry_run_stage_1_only_and_skip_tts(cache, monkeypatch, caplog):
    _rel(cache, (94,)).write_text("rel", encoding="utf-8")
    _main_dry_run(cache, monkeypatch, ["94", "--skip-tts"])
    assert "loaded from cache" in caplog.text and "skipped: --skip-tts" in caplog.text
    assert ".mp3" not in caplog.text
    caplog.clear()
    _main_dry_run(cache, monkeypatch, ["95", "--stage-1-only"])
    assert "not generated (--stage-1-only)" in caplog.text
    assert "stops after Stage 1" in caplog.text and "Stage 3 polish" not in caplog.text
