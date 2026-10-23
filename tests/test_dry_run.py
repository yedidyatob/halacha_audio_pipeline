"""main.py --dry-run: the run plan (resolve_run_plan, shared with a real run) without any API call."""
from unittest.mock import MagicMock, patch

import pytest

import main as main_mod
from pipeline.generator import GeminiScriptGenerator


def _cfg(tmp_path):
    return MagicMock(section_slug="YD", drafts_dir=str(tmp_path / "drafts"), relations_dir=str(tmp_path / "rel"),
                     output_dir=str(tmp_path / "out"), generator_max_retries=1, generator_engine="gemini", tts_engine="gemini")


def _args(**kw):
    defaults = dict(relations_file=None, draft_file={}, single_pass=False, overwrite_cache=False, limit_lessons=None, batch=False,
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


def test_plan_draft_file_overrides_cache_and_overwrite_cache(cache):
    mine = cache / "gemini3_draft.txt"
    mine.write_text("mine 94", encoding="utf-8")
    plan = main_mod.resolve_run_plan(_args(draft_file={94: str(mine)}, overwrite_cache=True), _cfg(cache),
                                     [94], [94, 95, 96], "m")
    assert plan.drafts == {94: "mine 94"} and plan.draft_files == [94]
    assert plan.draft_paths[94] == str(mine) and plan.missing_drafts == [95, 96]


def test_plan_draft_file_with_cached_others(cache):
    mine = cache / "d95.txt"
    mine.write_text("mine 95", encoding="utf-8")
    plan = main_mod.resolve_run_plan(_args(draft_file={95: str(mine)}), _cfg(cache), [94, 95, 96], [94, 95, 96], "m")
    assert plan.drafts == {94: "draft 94", 95: "mine 95", 96: "draft 96"} and plan.missing_drafts == []


@pytest.mark.parametrize("kw, message", [
    (dict(draft_file={94: "nope.txt"}), "Draft file for Siman 94 not found"),
    (dict(draft_file={95: "x.txt"}), r"does not need drafts for"),
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
    assert "Failed LLM calls are retried as new batch jobs: Stages 1-3 up to 1 time(s)\n" in out
    assert f"Stage 2 relations map: will be GENERATED (a batch job) from the drafts of [94, 95, 96] " \
           f"and saved to {_rel(cache)}" in out
    assert "from cache (2): [94, 96]" in out
    assert "to be GENERATED (1, as batch jobs, after a Sefaria fetch each): [95]" in out
    assert f"Siman 95 -> {cache / 'drafts' / 'YD_Siman_95_m_draft.txt'}" in out
    assert "Stage 3 polish and nikkud -> 3.1 dictionary nikkud: [94]" in out and "pronunciation" not in out
    base = cache / 'out' / 'YD_Siman_94_m'
    assert "Stage 3 polish from cache (0): none" in out
    assert f"Stage 3 polish to be GENERATED (1, as batch jobs): [94]\n" in out
    assert f"Siman 94 -> {base}_polish.txt" in out
    assert f"Siman 94: {base}_transcript.txt, {base}.mp3" in out


def test_main_dry_run_stage_1_only_and_skip_tts(cache, monkeypatch, caplog):
    _rel(cache, (94,)).write_text("rel", encoding="utf-8")
    _main_dry_run(cache, monkeypatch, ["94", "--skip-tts"])
    assert "loaded from cache" in caplog.text and "skipped: --skip-tts" in caplog.text
    assert ".mp3" not in caplog.text
    caplog.clear()
    _main_dry_run(cache, monkeypatch, ["95", "--stage-1-only"])
    assert "not generated (--stage-1-only)" in caplog.text
    assert "stops after Stage 1" in caplog.text and "Stage 3 polish" not in caplog.text


def test_parse_draft_file_arg():
    assert main_mod.parse_draft_file_arg("96=C:\\d\\x=y.txt") == (96, "C:\\d\\x=y.txt")
    for bad in ("96", "=p.txt", "96=", "94-96=p.txt"):
        with pytest.raises(main_mod.argparse.ArgumentTypeError):
            main_mod.parse_draft_file_arg(bad)


def test_draft_file_given_twice_is_rejected(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py", "94", "--draft-file", "94=a", "--draft-file", "94=b"])
    with pytest.raises(SystemExit) as exc:
        main_mod.parse_arguments()
    assert exc.value.code == 2


def test_dry_run_reports_draft_file(cache, monkeypatch, caplog):
    mine = cache / "d94.txt"
    mine.write_text("mine", encoding="utf-8")
    with caplog.at_level("INFO"):
        _main_dry_run(cache, monkeypatch, ["94", "--draft-file", f"94={mine}"])
    assert f"from --draft-file (1): [94]" in caplog.text and str(mine) in caplog.text
