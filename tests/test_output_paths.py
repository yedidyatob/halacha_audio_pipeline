"""Every saved output is logged once at INFO by save_output_file: '<label> saved: <latest> (history copy: <history>)'."""
from unittest.mock import MagicMock, patch

import pytest

import main as main_mod
from pipeline.generator import GeminiScriptGenerator
from pipeline.pronunciation import PronunciationVocalizer
from tests.test_stage3_flow import DRAFTS, FakeBatchGenerator, _config, _run, vocalized


def _saved_lines(caplog, prefix):
    return [r.getMessage() for r in caplog.records
            if r.levelname == "INFO" and r.getMessage().startswith(prefix)]


def _assert_saved_once(caplog, tmp_path, prefix, file_name):
    """Exactly one INFO line for the file: '<prefix> saved: <latest> (history copy: <history>)'."""
    lines = _saved_lines(caplog, prefix + " saved: ")
    assert len(lines) == 1, lines
    latest = tmp_path / file_name
    history = lines[0].split("(history copy: ")[1].rstrip(")")
    assert lines[0] == f"{prefix} saved: {latest} (history copy: {history})"
    assert latest.exists() and latest.read_text(encoding="utf-8") == open(history, encoding="utf-8").read()
    return latest


@pytest.mark.parametrize("batch", [True, False])
def test_stage3_saves_and_logs_polished_output_per_siman(tmp_path, caplog, batch):
    if batch:
        gen = FakeBatchGenerator(lambda cid, prompt: f"polished {prompt}")
    else:
        gen = MagicMock()
        gen.polish_siman_script.side_effect = lambda draft, rel, instr, **_: f"polished {draft}"
    with caplog.at_level("INFO"):
        summary = _run(tmp_path, gen, None, batch=batch)
    assert summary.succeeded == list(DRAFTS)
    for siman, draft in DRAFTS.items():
        latest = _assert_saved_once(caplog, tmp_path, f"Stage 3 (Polishing) output for Siman {siman}",
                                    f"YD_Siman_{siman}_m_polish.txt")
        assert latest.read_text(encoding="utf-8") == f"polished {draft}"  # before the nikkud stages


def test_stage3_6_logs_transcript_and_pronunciation_changes_paths(tmp_path, caplog):
    polish_gen = FakeBatchGenerator(lambda cid, prompt: prompt)
    voc_gen = FakeBatchGenerator(lambda cid, prompt: vocalized(prompt))
    with caplog.at_level("INFO"):
        _run(tmp_path, polish_gen, PronunciationVocalizer(voc_gen, "VOC", None, 0, [], 0.02), batch=True, drafts={94: "טית 94"})
    _assert_saved_once(caplog, tmp_path, "Stage 3.6 (Transcript save) output for Siman 94",
                       "YD_Siman_94_m_transcript.txt")
    _assert_saved_once(caplog, tmp_path, "Stage 3.5 (Pronunciation nikkud) changes for Siman 94",
                       "YD_Siman_94_m_pronunciation_changes.txt")


def test_tts_logs_audio_path_once(tmp_path, caplog):
    engine = MagicMock()
    engine.synthesize.side_effect = lambda text, output_path: open(output_path, "w", encoding="utf-8").write(text)
    with patch.object(main_mod, "create_tts_engine", return_value=engine), caplog.at_level("INFO"):
        main_mod.synthesize_siman_audio(94, "audio", _config(tmp_path), "m")
    _assert_saved_once(caplog, tmp_path, "TTS output for Siman 94", "YD_Siman_94_m.mp3")
    assert sum(str(tmp_path) in r.getMessage() for r in caplog.records) == 1  # the path is logged in one place


def test_full_run_logs_relations_polish_and_transcript_paths(tmp_path, monkeypatch, caplog):
    drafts_dir = tmp_path / "drafts"
    drafts_dir.mkdir()
    (drafts_dir / "YD_Siman_94_m_draft.txt").write_text("draft 94", encoding="utf-8")
    cfg = MagicMock(section_slug="YD", drafts_dir=str(drafts_dir), relations_dir=str(tmp_path),
                    output_dir=str(tmp_path), generator_max_retries=0, pronunciation_enabled=False)
    gen = MagicMock(spec=GeminiScriptGenerator, model_name="m")
    gen.analyze_cross_relations.return_value = "relations"
    gen.polish_siman_script.return_value = "polished 94"
    monkeypatch.setattr("sys.argv", ["main.py", "94", "--skip-tts"])
    with patch.object(main_mod, "PipelineConfig", return_value=cfg), \
         patch.object(main_mod, "create_generator_engine", return_value=gen), \
         patch.object(main_mod, "SefariaExtractor"), \
         caplog.at_level("INFO"):
        main_mod.main()
    _assert_saved_once(caplog, tmp_path, "Stage 2 (Relations) output for Simanim [94]", "YD_Relations_94_m.txt")
    _assert_saved_once(caplog, tmp_path, "Stage 3 (Polishing) output for Siman 94", "YD_Siman_94_m_polish.txt")
    _assert_saved_once(caplog, tmp_path, "Stage 3.6 (Transcript save) output for Siman 94",
                       "YD_Siman_94_m_transcript.txt")
    # the cached draft is loaded, not saved again
    assert not [r for r in caplog.records if r.getMessage().startswith("Stage 1 (Drafts) output")]
