"""scripts/polish_sample.py: polish a few seifim of a cached Stage 1 draft (generator mocked)."""
import os
from unittest.mock import MagicMock, patch

import pytest

from pipeline.gematria import gematria_to_int
from scripts import polish_sample as ps

DRAFT = """# סימן צ"ו

## סעיף א': חתיכה
פסק המחבר: א

## סעיף ב': סכין
פסק הרמ"א: ב

## סעיף ג׳: צנון
ג
"""


def test_gematria_to_int():
    assert [gematria_to_int(t) for t in ("א'", "ב", "ט\"ו", "טז", "צ״ו", "קכ\"ב")] == [1, 2, 15, 16, 96, 122]
    with pytest.raises(ValueError):
        gematria_to_int("abc")


def test_draft_slice_keeps_preamble_and_requested_seifim():
    out = ps.draft_slice(DRAFT, [1, 3])
    assert out.startswith('# סימן צ"ו') and "סעיף א'" in out and "סעיף ג׳" in out and "סכין" not in out


@pytest.mark.parametrize("draft, seifim, message", [
    ("no headings", [1], "No seif headings"),
    (DRAFT, [4], r"Seifim \[4\] are not in the draft \(it has \[1, 2, 3\]\)"),
    ("## סעיף א': x\n## סעיף א': y", [1], "appears twice"),
])
def test_draft_slice_errors(draft, seifim, message):
    with pytest.raises(ps.SampleError, match=message):
        ps.draft_slice(draft, seifim)


def _cfg(tmp_path):
    for d in ("drafts", "rel", "out"):
        (tmp_path / d).mkdir(exist_ok=True)
    return MagicMock(section_slug="YD", drafts_dir=str(tmp_path / "drafts"), relations_dir=str(tmp_path / "rel"),
                     output_dir=str(tmp_path / "out"), polishing_instruction="POLISH", polish_temperature=0.1)


def test_find_cached_relations_newest_including_siman(tmp_path):
    cfg = _cfg(tmp_path)
    rel = tmp_path / "rel"
    for name, mtime in (("YD_Relations_94_95_96_m.txt", 1), ("YD_Relations_96_97_m.txt", 2),
                        ("YD_Relations_96_97_m_20261022_101010.txt", 3), ("YD_Relations_94_95_m.txt", 4),
                        ("YD_Relations_96_other.txt", 5)):
        (rel / name).write_text("r", encoding="utf-8")
        os.utime(rel / name, (mtime, mtime))
    assert ps.find_cached_relations(cfg, 96, "m") == str(rel / "YD_Relations_96_97_m.txt")
    assert ps.find_cached_relations(cfg, 99, "m") is None


def _run(tmp_path, argv, gen=None):
    gen = gen or MagicMock(model_name="m")
    gen.polish_siman_script.return_value = "שיעור קצר מאוד"
    with patch.object(ps, "PipelineConfig", return_value=_cfg(tmp_path)), \
         patch.object(ps, "create_generator_engine", return_value=gen):
        path = ps.run(ps.parse_arguments(argv))
    return gen, path


def test_run_uses_cached_draft_and_relations(tmp_path, capsys):
    cfg = _cfg(tmp_path)
    (tmp_path / "drafts" / "YD_Siman_96_m_draft.txt").write_text(DRAFT, encoding="utf-8")
    (tmp_path / "rel" / "YD_Relations_95_96_m.txt").write_text("REL", encoding="utf-8")
    gen, path = _run(tmp_path, ["96", "--seifim", "1-2"])
    draft, relations, instruction = gen.polish_siman_script.call_args.args
    assert "חתיכה" in draft and "סכין" in draft and "צנון" not in draft
    assert (relations, instruction) == ("REL", "POLISH")
    assert gen.polish_siman_script.call_args.kwargs == {"temperature": 0.1}
    assert path == os.path.join(cfg.output_dir, "YD_Siman_96_m_sample_seifim_1_2.txt")
    assert open(path, encoding="utf-8").read() == "שיעור קצר מאוד"
    out = capsys.readouterr().out
    assert "YD_Relations_95_96_m.txt" in out and "(3 words; draft slice" in out
    assert os.listdir(tmp_path / "drafts") == ["YD_Siman_96_m_draft.txt"]  # no cache writes


def test_run_draft_file_and_relations_file(tmp_path, capsys):
    _cfg(tmp_path)
    mine, rel = tmp_path / "mine.txt", tmp_path / "myrel.txt"
    mine.write_text(DRAFT, encoding="utf-8")
    rel.write_text("MYREL", encoding="utf-8")
    (tmp_path / "rel" / "YD_Relations_96_m.txt").write_text("CACHED", encoding="utf-8")
    gen, _ = _run(tmp_path, ["96", "--seifim", "3", "--draft-file", str(mine), "--relations-file", str(rel)])
    assert gen.polish_siman_script.call_args.args[1] == "MYREL"
    assert f"Relations map: {rel}" in capsys.readouterr().out


def test_run_without_relations_map_stops_before_polish(tmp_path):
    _cfg(tmp_path)
    (tmp_path / "drafts" / "YD_Siman_96_m_draft.txt").write_text(DRAFT, encoding="utf-8")
    gen = MagicMock(model_name="m")
    with pytest.raises(ps.SampleError, match="No relations map covers Siman 96.*--relations-file PATH.*"
                                             "--allow-no-relations"):
        _run(tmp_path, ["96", "--seifim", "1"], gen)
    gen.polish_siman_script.assert_not_called()


def test_run_allow_no_relations(tmp_path, capsys):
    _cfg(tmp_path)
    (tmp_path / "drafts" / "YD_Siman_96_m_draft.txt").write_text(DRAFT, encoding="utf-8")
    gen, _ = _run(tmp_path, ["96", "--seifim", "3", "--allow-no-relations"])
    assert gen.polish_siman_script.call_args.args[1] == ""
    assert "Relations map: none" in capsys.readouterr().out


def test_run_missing_draft_never_generates(tmp_path):
    gen = MagicMock(model_name="m")
    with pytest.raises(ps.SampleError, match="Stage 1 draft for Siman 96 not found"):
        _run(tmp_path, ["96", "--seifim", "1"], gen)
    assert gen.method_calls == []


@pytest.mark.parametrize("argv", [["94-95", "--seifim", "1"], ["96", "--seifim", "x"], ["96"]])
def test_parse_arguments_errors(argv):
    with pytest.raises(SystemExit):
        ps.parse_arguments(argv)
