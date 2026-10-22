"""Stage 3 polish cache: an existing {section}_Siman_{N}_{model}_polish.txt is loaded instead of polishing again."""
from unittest.mock import MagicMock, patch

import pytest

import main as main_mod
from pipeline.generator import GeminiScriptGenerator
from tests.test_dry_run import _args as _plan_args, _main_dry_run
from tests.test_stage3_flow import DRAFTS, FakeBatchGenerator, _run, _transcripts


def _polish_file(directory, siman):
    return directory / f"YD_Siman_{siman}_m_polish.txt"


def _plan_cfg(tmp_path):
    return MagicMock(section_slug="YD", drafts_dir=str(tmp_path / "drafts"), relations_dir=str(tmp_path / "rel"),
                     output_dir=str(tmp_path / "out"), generator_max_retries=1, generator_engine="gemini",
                     tts_engine="gemini")


@pytest.fixture
def out(tmp_path):
    for name in ("drafts", "rel", "out"):
        (tmp_path / name).mkdir()
    for s in (94, 95, 96):
        (tmp_path / "drafts" / f"YD_Siman_{s}_m_draft.txt").write_text(f"draft {s}", encoding="utf-8")
    _polish_file(tmp_path / "out", 94).write_text("cached 94", encoding="utf-8")
    _polish_file(tmp_path / "out", 96).write_text(" \n\t", encoding="utf-8")  # whitespace only
    return tmp_path


# --- plan ------------------------------------------------------------------------

def test_plan_loads_cached_polish_and_logs_its_path(out, caplog):
    with caplog.at_level("INFO"):
        plan = main_mod.resolve_run_plan(_plan_args(), _plan_cfg(out), [94, 95, 96], [94, 95, 96], "m")
    assert plan.polished == {94: "cached 94"}
    assert plan.missing_polish == [95, 96]
    assert plan.polish_paths == {s: str(_polish_file(out / "out", s)) for s in (94, 95, 96)}
    assert f"Loaded cached Stage 3 polish for Siman 94: {_polish_file(out / 'out', 94)} (9 chars)." in caplog.messages
    assert f"Cached Stage 3 polish for Siman 96 is empty: {_polish_file(out / 'out', 96)}. Will regenerate." \
        in caplog.messages


def test_plan_overwrite_cache_regenerates_polish(out):
    plan = main_mod.resolve_run_plan(_plan_args(overwrite_cache=True), _plan_cfg(out), [94, 95], [94, 95], "m")
    assert plan.polished == {} and plan.missing_polish == [94, 95]


def test_plan_polish_only_for_target_simanim(out):
    plan = main_mod.resolve_run_plan(_plan_args(limit_lessons=1), _plan_cfg(out), [95, 94], [94, 95, 96], "m")
    assert list(plan.polish_paths) == [95] and plan.polished == {} and plan.missing_polish == [95]


def test_dry_run_shows_polish_per_siman(out, monkeypatch, caplog):
    gen = _main_dry_run(out, monkeypatch, ["94-96", "--skip-tts"])  # same layout: drafts/, rel/, out/
    text = caplog.text
    assert gen.method_calls == []
    assert "Stage 3 polish from cache (1): [94]" in text
    assert f"    Siman 94: {_polish_file(out / 'out', 94)}\n" in text
    assert "Stage 3 polish to be GENERATED (2, as synchronous calls): [95, 96]" in text
    assert f"    Siman 95 -> {_polish_file(out / 'out', 95)}\n" in text
    assert f"    Siman 96 -> {_polish_file(out / 'out', 96)}\n" in text


# --- Stage 3 flow: sync and batch ------------------------------------------------

def test_sync_cache_hit_skips_polish_call(tmp_path):
    gen = MagicMock()
    gen.polish_siman_script.side_effect = lambda draft, rel, instr, **_: f"polished {draft}"
    summary = _run(tmp_path, gen, batch=False, cached_polish={94: "cached טית 94", 96: "cached 96"})
    assert summary.succeeded == list(DRAFTS)
    assert [c.args[0] for c in gen.polish_siman_script.call_args_list] == [DRAFTS[95], DRAFTS[97]]
    # the cached polish goes on through the later stages; only the new ones are saved as polish
    assert (tmp_path / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8") == "cached טית 94"
    assert not _polish_file(tmp_path, 94).exists() and _polish_file(tmp_path, 95).exists()


def test_sync_cache_hit_needs_no_draft(tmp_path):
    gen = MagicMock()
    summary = _run(tmp_path, gen, batch=False, drafts={}, targets=[94], cached_polish={94: "cached 94"})
    assert summary.succeeded == [94] and not summary.failed
    gen.polish_siman_script.assert_not_called()


def test_batch_polishes_only_uncached_simanim(tmp_path):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    summary = _run(tmp_path, polish_gen, batch=True,
                   cached_polish={95: "טֵית cached 95", 97: "טֵית cached 97"})
    assert summary.succeeded == list(DRAFTS)
    assert [r["custom_id"] for r in polish_gen.submitted] == ["stage3_siman_94", "stage3_siman_96"]
    # the cached (already vocalized) polish goes on to the dictionary stage and the transcript as is
    assert (tmp_path / "YD_Siman_95_m_transcript.txt").read_text(encoding="utf-8") == "טֵית cached 95"


def test_batch_all_cached_submits_no_polish_job(tmp_path, caplog):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    cached = {s: f"cached {d}" for s, d in DRAFTS.items()}
    with caplog.at_level("INFO"):
        summary = _run(tmp_path, polish_gen, batch=True,
                       cached_polish=cached)
    assert summary.succeeded == list(DRAFTS) and polish_gen.submitted == []
    assert "Stage 3 (Polishing and nikkud): no calls needed; every target siman's polish was loaded from cache." in caplog.messages
    assert not [m for m in caplog.messages if "Stage 3 (Polishing and nikkud) batch" in m]
    assert _transcripts(tmp_path) == list(DRAFTS)


# --- full run --------------------------------------------------------------------

def _main(tmp_path, monkeypatch, argv, gen):
    cfg = MagicMock(section_slug="YD", drafts_dir=str(tmp_path / "drafts"), relations_dir=str(tmp_path / "rel"),
                    output_dir=str(tmp_path / "out"), generator_max_retries=0)
    monkeypatch.setattr("sys.argv", ["main.py", *argv, "--skip-tts"])
    with patch.object(main_mod, "PipelineConfig", return_value=cfg), \
         patch.object(main_mod, "create_generator_engine", return_value=gen), \
         patch.object(main_mod, "SefariaExtractor"):
        main_mod.main()


def _cached_run_dir(tmp_path):
    for name in ("drafts", "rel", "out"):
        (tmp_path / name).mkdir()
    (tmp_path / "drafts" / "YD_Siman_94_m_draft.txt").write_text("draft 94", encoding="utf-8")
    (tmp_path / "rel" / "YD_Relations_94_m.txt").write_text("rel", encoding="utf-8")
    _polish_file(tmp_path / "out", 94).write_text("cached 94", encoding="utf-8")


def test_full_run_with_cached_polish_makes_no_generator_call(tmp_path, monkeypatch):
    _cached_run_dir(tmp_path)
    gen = MagicMock(spec=GeminiScriptGenerator, model_name="m")
    _main(tmp_path, monkeypatch, ["94"], gen)
    assert gen.method_calls == []
    assert (tmp_path / "out" / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8") == "cached 94"


def test_full_run_overwrite_cache_regenerates_polish(tmp_path, monkeypatch):
    _cached_run_dir(tmp_path)
    gen = MagicMock(spec=GeminiScriptGenerator, model_name="m")
    gen.generate_siman_script.return_value = "new draft"
    gen.analyze_cross_relations.return_value = "new rel"
    gen.polish_siman_script.return_value = "new polish"
    _main(tmp_path, monkeypatch, ["94", "--overwrite-cache"], gen)
    assert gen.polish_siman_script.call_args.args[:2] == ("new draft", "new rel")
    assert _polish_file(tmp_path / "out", 94).read_text(encoding="utf-8") == "new polish"
    assert (tmp_path / "out" / "YD_Siman_94_m_transcript.txt").read_text(encoding="utf-8") == "new polish"
