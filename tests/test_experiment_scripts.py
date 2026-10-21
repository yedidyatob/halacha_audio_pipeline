"""Experiment scripts: combined polish + nikkud in one call, and polish only."""
import os
from unittest.mock import MagicMock

import pytest

from scripts.polish_and_vocalize_combined import (
    build_combined_instruction, pronunciation_rules, validate_combined_output,
)
from pipeline.pronunciation import CHANGES_HEADER, Change, PronunciationError

PRON = (
    "אתה עורך הגייה.\n"
    "החזר את התסריט בדיוק כפי שהוא, מילה במילה.\n"
    "השינוי היחיד המותר: להוסיף ניקוד.\n"
    "\n"
    "נקד מילה רק אם היא דו-משמעית.\n"
    "\n"
    "תמיד לנקד: כותח\n"
)
TET = "טֵית"


# --- prompt assembly ---------------------------------------------------------------

def test_conflicting_first_paragraph_is_dropped():
    rules = pronunciation_rules(PRON)
    assert rules == "נקד מילה רק אם היא דו-משמעית.\n\nתמיד לנקד: כותח"
    assert "בדיוק כפי שהוא" not in rules and "השינוי היחיד המותר" not in rules


def test_unexpected_instruction_structure_is_refused():
    with pytest.raises(ValueError, match="first|paragraph"):
        pronunciation_rules("החזר את התסריט בדיוק כפי שהוא.\nנקד מילים.")  # no blank line
    with pytest.raises(ValueError):
        pronunciation_rules("פסקה בלי כלל העתקה\n\nכללים")  # first paragraph lacks the exact-copy rule


def test_combined_instruction_order_and_separation():
    out = build_combined_instruction("POLISH RULES\n", "===== HEADER =====\n", PRON)
    assert out == "POLISH RULES\n\n===== HEADER =====\nנקד מילה רק אם היא דו-משמעית.\n\nתמיד לנקד: כותח\n"


def test_real_config_combined_instruction():
    from unittest.mock import patch
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        cfg = PipelineConfig("config.yaml")
    out = build_combined_instruction(cfg.polishing_instruction, cfg.experiment_combined_addendum_header,
                                     cfg.pronunciation_instruction)
    assert out.startswith(cfg.polishing_instruction.rstrip())
    assert "תוספת: ניקוד לצורך הגייה" in out and "<" not in out
    assert "החזר את התסריט בדיוק כפי שהוא" not in out
    assert "{always_vocalize}" not in out and "תמיד לנקד" in out
    assert "כחל (כְּחָל = עטין של בהמה; כֹּחַל = צבע איפור)" in out  # readings rendered as in Stage 3.5


# --- validation and the script ------------------------------------------------------

def test_validate_combined_output():
    assert validate_combined_output(f"שיעור חדש על {TET}.") == [Change("טית", TET, 1, f"שיעור חדש על [{TET}].")]
    for bad in ('שיעור <phoneme alphabet="ipa" ph="tet">טית</phoneme>', "שיעור <break time=\"1s\"/> חדש"):
        with pytest.raises(PronunciationError, match="markup"):
            validate_combined_output(bad)
    with pytest.raises(PronunciationError, match="empty"):
        validate_combined_output("  ")


def _run_combined(tmp_path, monkeypatch, llm_text, extra_args=()):
    import scripts.polish_and_vocalize_combined as script
    monkeypatch.chdir(tmp_path)
    draft = tmp_path / "YD_Siman_94_m_draft.txt"
    relations = tmp_path / "YD_Relations_94_m.txt"
    draft.write_text("טיוטה", encoding="utf-8")
    relations.write_text("קשרים", encoding="utf-8")
    gen = MagicMock(model_name="m", supports_thinking_level=True)
    gen.polish_siman_script.return_value = llm_text
    cfg = MagicMock(polishing_instruction="POLISH", experiment_combined_addendum_header="HEADER",
                    pronunciation_instruction=PRON, pronunciation_thinking_level="low", generator_engine="gemini")
    monkeypatch.setattr("pipeline.config.PipelineConfig", lambda path: cfg)
    monkeypatch.setattr("pipeline.factory.create_generator_engine", lambda c: gen)
    script.main([str(draft), str(relations), *extra_args])
    return gen, draft


def test_combined_script_writes_outputs(tmp_path, monkeypatch, capsys):
    gen, draft = _run_combined(tmp_path, monkeypatch, f'שיעור על {TET} והש"ך.')
    call = gen.polish_siman_script.call_args
    assert call.args[:2] == ("טיוטה", "קשרים") and call.kwargs == {"thinking_level": "low"}
    assert call.args[2].startswith("POLISH\n\nHEADER\n") and "בדיוק כפי שהוא" not in call.args[2]
    out = tmp_path / "YD_Siman_94_m_draft_combined_vocalized.txt"
    assert out.read_text(encoding="utf-8") == f"שיעור על {TET} והשַׁךְ."  # dictionary nikkud applied after
    # only the LLM's own vocalization is listed (before the dictionary pass)
    assert (tmp_path / "YD_Siman_94_m_draft_combined_vocalized_changes.txt").read_text(encoding="utf-8") == (
        f"{CHANGES_HEADER}\nno\tטית\t{TET}\t1\t1\tשיעור על [{TET}] והש\"ך.\n")
    assert "thinking level: low" in capsys.readouterr().out


def test_combined_script_leftover_markup_exit_1(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_combined(tmp_path, monkeypatch, 'שיעור <phoneme alphabet="ipa">טית</phoneme>')
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "markup in output" in captured.err
    assert sorted(os.listdir(tmp_path)) == ["YD_Relations_94_m.txt", "YD_Siman_94_m_draft.txt"]


def test_polish_draft_script_uses_pipeline_polish(tmp_path, monkeypatch):
    import scripts.polish_draft as script
    monkeypatch.chdir(tmp_path)
    (tmp_path / "d_draft.txt").write_text("טיוטה", encoding="utf-8")
    (tmp_path / "r.txt").write_text("קשרים", encoding="utf-8")
    gen = MagicMock(model_name="m")
    gen.polish_siman_script.return_value = "מלוטש"
    cfg = MagicMock(polishing_instruction="POLISH", generator_engine="gemini")
    monkeypatch.setattr("pipeline.config.PipelineConfig", lambda path: cfg)
    monkeypatch.setattr("pipeline.factory.create_generator_engine", lambda c: gen)
    script.main([str(tmp_path / "d_draft.txt"), str(tmp_path / "r.txt")])
    gen.polish_siman_script.assert_called_once_with("טיוטה", "קשרים", "POLISH")
    assert (tmp_path / "d_draft_polished.txt").read_text(encoding="utf-8") == "מלוטש"


def test_gemini_polish_passes_thinking_level_and_logs_cost(caplog):
    """The combined call is the pipeline's polish request (temperature 0.1) plus a thinking_level;
    its usage and cost are logged from gemini_pricing like every Gemini call."""
    from unittest.mock import patch
    from types import SimpleNamespace
    from pipeline.generator import GeminiScriptGenerator
    with patch("pipeline.generator.build_gemini_client"):
        gen = GeminiScriptGenerator(api_key="k", model_name="m", temperature=0.3, ssl_verify=True,
                                    pricing={"input_per_million": 2.0, "output_per_million": 12.0},
                                    thinking_dir="t")
    usage = SimpleNamespace(prompt_token_count=1000, candidates_token_count=2000, thoughts_token_count=500,
                            total_token_count=3500)
    response = MagicMock(usage_metadata=usage)
    with patch("pipeline.generator._collect_stream", return_value=(response, "")), \
         patch("pipeline.generator._gemini_response_text", return_value="out"), caplog.at_level("INFO"):
        assert gen.polish_siman_script("draft", "rel", "COMBINED", thinking_level="high") == "out"
    config = gen.client.models.generate_content_stream.call_args.kwargs["config"]
    assert config.system_instruction == "COMBINED" and config.temperature == 0.1
    assert config.thinking_config.thinking_level.lower() == "high"
    assert gen.client.models.generate_content_stream.call_args.kwargs["contents"] == \
        gen._get_polishing_user_prompt("draft", "rel")
    assert "Estimated Cost: $0.032000 USD" in caplog.text  # 1000*$2 + (2000+500)*$12, per 1M


def test_thinking_level_support_flags():
    from pipeline.generator import GeminiScriptGenerator, OpenAIScriptGenerator
    assert GeminiScriptGenerator.supports_thinking_level is True
    assert OpenAIScriptGenerator.supports_thinking_level is False
