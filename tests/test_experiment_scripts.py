"""Experiment scripts: combined polish + nikkud, polish-only, and usage/cost reporting."""
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pipeline.usage import ModelPricing, TextResult, TokenUsage, estimate_cost_usd, format_usage_report
from scripts.polish_and_vocalize_combined import (
    build_combined_instruction, pronunciation_rules, validate_combined_output,
)
from pipeline.pronunciation import PronunciationError

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
PRICING = ModelPricing(input_per_1m_usd=2.0, output_per_1m_usd=12.0, max_prompt_tokens=200_000)


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
    assert "{always_vocalize}" not in out and "תמיד לנקד:" in out
    assert cfg.experiment_pricing("gemini-3.1-pro-preview") == PRICING


# --- usage / cost ------------------------------------------------------------------

def test_cost_math_bills_thinking_at_output_rate():
    usage = TokenUsage(prompt_tokens=100_000, output_tokens=10_000, thinking_tokens=5_000)
    assert estimate_cost_usd(usage, PRICING) == pytest.approx(0.2 + 0.18)  # 100k*$2 + 15k*$12, per 1M


def test_usage_from_gemini_and_openai():
    gem = SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20,
                                                         thoughts_token_count=None))
    assert TokenUsage.from_gemini(gem) == TokenUsage(10, 20, 0)
    oai = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=30,
                                                completion_tokens_details=SimpleNamespace(reasoning_tokens=12)))
    assert TokenUsage.from_openai(oai) == TokenUsage(7, 18, 12)  # completion includes reasoning
    assert TokenUsage.from_gemini(SimpleNamespace()) == TokenUsage(0, 0, 0)


def test_usage_report_format_and_tier_warning():
    report = format_usage_report("Polish", "m", TokenUsage(1_000, 2_000, 3_000), PRICING)
    assert "Prompt tokens:   1,000" in report and "Thinking tokens: 3,000" in report
    assert "Estimated cost:  $0.0620" in report and "WARNING" not in report
    assert "WARNING" in format_usage_report("Polish", "m", TokenUsage(200_001, 0, 0), PRICING)


def test_config_pricing_missing_or_invalid():
    from tests.test_pronunciation import _config
    with pytest.raises(ValueError, match="experiment_pricing.x"):
        _config().experiment_pricing("x")
    cfg = _config('experiment_pricing:\n  x:\n    input_per_1m_usd: 1\n    output_per_1m_usd: "a"\n    max_prompt_tokens: 5\n')
    with pytest.raises(ValueError, match="output_per_1m_usd"):
        cfg.experiment_pricing("x")


# --- validation and the script ------------------------------------------------------

def test_validate_combined_output():
    assert validate_combined_output(f"שיעור חדש על {TET}.") == [("טית.", f"{TET}.")]
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
    gen.polish.return_value = TextResult(llm_text, TokenUsage(1000, 2000, 500))
    cfg = MagicMock(polishing_instruction="POLISH", experiment_combined_addendum_header="HEADER",
                    pronunciation_instruction=PRON, pronunciation_thinking_level="low", generator_engine="gemini")
    cfg.experiment_pricing.return_value = PRICING
    monkeypatch.setattr("pipeline.config.PipelineConfig", lambda path: cfg)
    monkeypatch.setattr("pipeline.factory.create_generator_engine", lambda c: gen)
    script.main([str(draft), str(relations), *extra_args])
    return gen, draft


def test_combined_script_writes_outputs_and_usage(tmp_path, monkeypatch, capsys):
    gen, draft = _run_combined(tmp_path, monkeypatch, f'שיעור על {TET} והש"ך.')
    args = gen.polish.call_args.args
    assert args[:2] == ("טיוטה", "קשרים") and args[3] == "low"
    assert args[2].startswith("POLISH\n\nHEADER\n") and "בדיוק כפי שהוא" not in args[2]
    out = tmp_path / "YD_Siman_94_m_draft_combined_vocalized.txt"
    assert out.read_text(encoding="utf-8") == f"שיעור על {TET} והשַׁךְ."  # dictionary nikkud applied after
    # only the LLM's own vocalization is listed (before the dictionary pass)
    assert (tmp_path / "YD_Siman_94_m_draft_combined_vocalized_changes.txt").read_text(encoding="utf-8") == f"טית\t{TET}\n"
    printed = capsys.readouterr().out
    assert "thinking level: low" in printed and "Estimated cost:  $0.0320" in printed


def test_combined_script_leftover_markup_exit_1(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_combined(tmp_path, monkeypatch, 'שיעור <phoneme alphabet="ipa">טית</phoneme>')
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "markup in output" in captured.err
    assert "Estimated cost" in captured.out  # the paid call's usage is still reported
    assert sorted(os.listdir(tmp_path)) == ["YD_Relations_94_m.txt", "YD_Siman_94_m_draft.txt"]


def test_polish_draft_script_uses_pipeline_polish_and_prints_usage(tmp_path, monkeypatch, capsys):
    import scripts.polish_draft as script
    monkeypatch.chdir(tmp_path)
    (tmp_path / "d_draft.txt").write_text("טיוטה", encoding="utf-8")
    (tmp_path / "r.txt").write_text("קשרים", encoding="utf-8")
    gen = MagicMock(model_name="m")
    gen.polish.return_value = TextResult("מלוטש", TokenUsage(10, 20, 0))
    cfg = MagicMock(polishing_instruction="POLISH", generator_engine="gemini")
    cfg.experiment_pricing.return_value = PRICING
    monkeypatch.setattr("pipeline.config.PipelineConfig", lambda path: cfg)
    monkeypatch.setattr("pipeline.factory.create_generator_engine", lambda c: gen)
    script.main([str(tmp_path / "d_draft.txt"), str(tmp_path / "r.txt")])
    gen.polish.assert_called_once_with("טיוטה", "קשרים", "POLISH")
    assert (tmp_path / "d_draft_polished.txt").read_text(encoding="utf-8") == "מלוטש"
    assert "Prompt tokens:   10" in capsys.readouterr().out


def test_base_polish_builds_the_pipeline_prompt():
    from pipeline.generator import POLISH_TEMPERATURE, GeminiScriptGenerator
    from unittest.mock import patch
    with patch("pipeline.generator.genai.Client"):
        gen = GeminiScriptGenerator(api_key="k", model_name="m", temperature=0.3)
    with patch.object(gen, "complete", return_value=TextResult("out", TokenUsage(0, 0, 0))) as complete:
        assert gen.polish_siman_script("draft", "rel", "INSTR") == "out"
        gen.polish("draft", "rel", "COMBINED", "high")
    first, second = complete.call_args_list
    assert first.args == ("INSTR", gen._get_polishing_user_prompt("draft", "rel"), POLISH_TEMPERATURE, None)
    assert second.args == ("COMBINED", gen._get_polishing_user_prompt("draft", "rel"), POLISH_TEMPERATURE, "high")


def test_vocalize_pronunciation_script_prints_usage(tmp_path, monkeypatch, capsys):
    import scripts.vocalize_pronunciation as script
    from pipeline.pronunciation import PronunciationVocalizer
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "x.txt"
    src.write_text("סימן טית", encoding="utf-8")
    gen = MagicMock(model_name="m", temperature=0.2)
    gen.complete.return_value = TextResult(f"סימן {TET}", TokenUsage(100, 50, 25))
    cfg = MagicMock(pronunciation_engine="gemini")
    cfg.experiment_pricing.return_value = PRICING
    monkeypatch.setattr("pipeline.config.PipelineConfig", lambda path: cfg)
    monkeypatch.setattr("pipeline.pronunciation.create_pronunciation_vocalizer",
                        lambda c: PronunciationVocalizer(gen, "INSTR", "high"))
    script.main([str(src)])
    printed = capsys.readouterr().out
    assert "Pronunciation nikkud: token usage (m)" in printed and "Thinking tokens: 25" in printed


def test_thinking_level_support_flags():
    from pipeline.generator import GeminiScriptGenerator, OpenAIScriptGenerator
    assert GeminiScriptGenerator.supports_thinking_level is True
    assert OpenAIScriptGenerator.supports_thinking_level is False
