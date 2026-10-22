"""Synchronous Gemini calls: streamed, thought summaries to a separate file, cost incl. thinking tokens."""
import logging
import os
from unittest.mock import MagicMock, mock_open, patch

import pytest
from google.genai import types

from pipeline.enhancer import ElevenLabsEnhancer
from pipeline.utils import gemini_cost, log_gemini_usage
from tests.gemini_fakes import PRICE, answer_stream, chunk, make_gemini_generator, usage


def _thinking_files(directory):
    return sorted(p for p in os.listdir(directory)) if os.path.isdir(directory) else []


# --- the shared streaming path ------------------------------------------------------

@pytest.mark.parametrize("call", [
    lambda g: g.generate_siman_script(94, "ctx", "sys", thinking_name="N"),
    lambda g: g.analyze_cross_relations({94: "d"}, "sys", thinking_name="N"),
    lambda g: g.polish_siman_script("d", "rel", "sys", temperature=0.1, thinking_name="N"),
    lambda g: g.generate_text("sys", "user", thinking_name="N"),
])
def test_every_sync_call_streams_and_returns_only_answer_text(tmp_path, call):
    gen, client = make_gemini_generator(tmp_path)
    client.models.generate_content_stream.return_value = answer_stream("התשובה", thought="SECRET THOUGHT")
    assert call(gen) == "התשובה"
    client.models.generate_content.assert_not_called()
    assert client.models.generate_content_stream.call_args.kwargs["config"].thinking_config.include_thoughts is True
    with open(tmp_path / "N_thinking.txt", encoding="utf-8") as f:
        assert f.read() == "SECRET THOUGHT"


def test_thoughts_never_leak_into_answer_even_when_interleaved(tmp_path):
    gen, client = make_gemini_generator(tmp_path)
    client.models.generate_content_stream.return_value = [
        chunk(("thought", "T1 ")),
        chunk(("text", "A"), ("thought", "T2")),
        chunk(("text", "B"), finish_reason=types.FinishReason.STOP, usage=usage()),
    ]
    assert gen.generate_text("sys", "user", thinking_name="X") == "AB"
    assert (tmp_path / "X_thinking.txt").read_text(encoding="utf-8") == "T1 T2"
    assert len(_thinking_files(tmp_path)) == 2  # timestamped history copy + latest copy


def test_thinking_file_path_logged_once(tmp_path, caplog):
    gen, client = make_gemini_generator(tmp_path)
    client.models.generate_content_stream.return_value = answer_stream("A", thought="why")
    with caplog.at_level("INFO"):
        gen.generate_siman_script(94, "ctx", "sys", thinking_name="N")
    lines = [r.getMessage() for r in caplog.records if " saved: " in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].startswith(f"Thinking summary of Gemini script generation for Siman 94 saved: "
                               f"{tmp_path / 'N_thinking.txt'} (history copy: {tmp_path / 'N_thinking_'}")


def test_no_thinking_file_without_name_or_without_thoughts(tmp_path):
    gen, client = make_gemini_generator(tmp_path / "th")
    client.models.generate_content_stream.return_value = answer_stream("a", thought="t")
    gen.generate_text("sys", "user")  # no thinking_name
    client.models.generate_content_stream.return_value = answer_stream("a", thought="")
    gen.generate_text("sys", "user", thinking_name="N")  # no thoughts in the stream
    assert _thinking_files(tmp_path / "th") == []


def test_thinking_level_sent_with_include_thoughts(tmp_path):
    gen, client = make_gemini_generator(tmp_path)
    client.models.generate_content_stream.return_value = answer_stream()
    gen.generate_siman_script(94, "ctx", "sys", thinking_level="medium")
    thinking = client.models.generate_content_stream.call_args.kwargs["config"].thinking_config
    assert thinking.include_thoughts is True and thinking.thinking_level.value.lower() == "medium"


def test_prompt_blocked_raises(tmp_path):
    gen, client = make_gemini_generator(tmp_path)
    client.models.generate_content_stream.return_value = [chunk(prompt_feedback=types.GenerateContentResponsePromptFeedback(
        block_reason=types.BlockedReason.SAFETY, block_reason_message="nope"))]
    with pytest.raises(ValueError, match=r"prompt blocked \(SAFETY\): nope"):
        gen.generate_text("sys", "user")


def test_early_finish_discards_partial_text_but_keeps_thinking_file(tmp_path):
    gen, client = make_gemini_generator(tmp_path)
    client.models.generate_content_stream.return_value = [
        chunk(("thought", "long thinking")),
        chunk(("text", "partial"), finish_reason=types.FinishReason.MAX_TOKENS, usage=usage()),
    ]
    with pytest.raises(ValueError, match=r"finish_reason MAX_TOKENS.*7 chars of partial text discarded"):
        gen.polish_siman_script("d", "", "sys", temperature=0.1, thinking_name="P")
    assert (tmp_path / "P_thinking.txt").read_text(encoding="utf-8") == "long thinking"


def test_thoughts_only_response_is_an_error(tmp_path):
    gen, client = make_gemini_generator(tmp_path)
    client.models.generate_content_stream.return_value = [
        chunk(("thought", "hmm"), finish_reason=types.FinishReason.STOP, usage=usage())]
    with pytest.raises(ValueError, match="response has no text"):
        gen.generate_text("sys", "user")


def test_empty_stream_is_an_error(tmp_path):
    gen, client = make_gemini_generator(tmp_path)
    client.models.generate_content_stream.return_value = []
    with pytest.raises(ValueError, match="no candidates"):
        gen.generate_text("sys", "user")


def test_usage_from_final_chunk_is_logged_with_thinking_cost(tmp_path, caplog):
    gen, client = make_gemini_generator(tmp_path, model_name="gemini-3.1-pro-preview")
    client.models.generate_content_stream.return_value = [
        chunk(("thought", "t"), usage=usage(prompt=1000, output=0, thoughts=10)),  # partial counts
        chunk(("text", "a"), finish_reason=types.FinishReason.STOP,
              usage=usage(prompt=1_000_000, output=100_000, thoughts=400_000)),
    ]
    with caplog.at_level(logging.INFO):
        gen.generate_text("sys", "user")
    line = next(r.getMessage() for r in caplog.records if "Cost Tracker" in r.getMessage())
    assert "Input=1000000, Output=100000, Thinking=400000" in line
    # long-context tier (prompt > 200k): 1,000,000 * $4/1M + (100,000 + 400,000 thinking) * $18/1M = 4 + 9
    assert "Estimated Cost: $13.000000 USD" in line


# --- cost ---------------------------------------------------------------------------

def test_gemini_cost_bills_thinking_as_output():
    flat = {"input_per_million": 2.0, "output_per_million": 12.0}
    assert gemini_cost(100_000, 50_000, flat) == pytest.approx(0.2 + 0.6)


def test_gemini_cost_long_context_tier():
    assert gemini_cost(200_000, 1_000_000, PRICE) == pytest.approx(0.4 + 12.0)   # at threshold: base tier
    assert gemini_cost(200_001, 1_000_000, PRICE) == pytest.approx(200_001 * 4 / 1e6 + 18.0)


def test_log_gemini_usage_includes_thoughts(caplog):
    logger = logging.getLogger("cost-test")
    response = MagicMock(usage_metadata=usage(prompt=10_000, output=1_000, thoughts=9_000))
    with caplog.at_level(logging.INFO, logger="cost-test"):
        log_gemini_usage(logger, response, "m", {"input_per_million": 2.0, "output_per_million": 12.0})
    # 10k*2 + (1k+9k)*12 = 0.02 + 0.12
    assert "Estimated Cost: $0.140000 USD" in caplog.text and "Thinking=9000" in caplog.text


def test_log_gemini_usage_without_usage_is_silent(caplog):
    log_gemini_usage(logging.getLogger("x"), MagicMock(usage_metadata=None), "m", PRICE)
    assert "Cost Tracker" not in caplog.text


# --- enhancer -----------------------------------------------------------------------

def test_enhancer_goes_through_shared_generator(tmp_path):
    gen, client = make_gemini_generator(tmp_path, model_name="gemini-3.1-flash-lite", ssl_verify=False)
    client.models.generate_content_stream.return_value = answer_stream("[warm] טקסט", thought="why")
    enhancer = ElevenLabsEnhancer(gen)
    path = enhancer.enhance_and_save("טקסט", str(tmp_path / "out"), "T_enhanced")
    assert open(path, encoding="utf-8").read() == "[warm] טקסט"
    assert (tmp_path / "T_enhanced_thinking.txt").read_text(encoding="utf-8") == "why"


def test_enhancer_client_honours_ssl_verify(tmp_path):
    with patch("pipeline.generator.genai.Client") as client_cls:
        from pipeline.generator import GeminiScriptGenerator
        GeminiScriptGenerator(api_key="k", model_name="m", temperature=0.2, ssl_verify=False,
                              pricing=PRICE, thinking_dir=str(tmp_path))
    assert client_cls.call_args.kwargs["http_options"] == {"client_args": {"verify": False}}


# --- config -------------------------------------------------------------------------

BASE = """halachic_section: "Yoreh De'ah"
directories:
  thinking_dir: "./out/thinking"
tts:
  engine: "elevenlabs"
  chunk_gap_paragraph_ms: 0
  chunk_gap_sentence_ms: 0
  elevenlabs:
    chunk_size: 8000
    context_chars: 500
"""


def _config(yaml_text):
    from pipeline.config import PipelineConfig
    with patch("builtins.open", mock_open(read_data=yaml_text)), \
         patch("os.path.exists", return_value=True), patch("os.makedirs"):
        return PipelineConfig("config.yaml")


def test_thinking_dir_required():
    with pytest.raises(ValueError, match="'thinking_dir' under 'directories'"):
        _config(BASE.replace('directories:\n  thinking_dir: "./out/thinking"\n', ""))
    assert _config(BASE).thinking_dir == "./out/thinking"


def test_draft_thinking_level():
    assert _config(BASE + 'generator:\n  engine: "gemini"\n  draft_thinking_level: "high"\n').draft_thinking_level == "high"
    with pytest.raises(ValueError, match="'draft_thinking_level' under 'generator'"):
        _config(BASE + 'generator:\n  engine: "gemini"\n').draft_thinking_level
    with pytest.raises(ValueError, match="low, medium, high"):
        _config(BASE + 'generator:\n  engine: "gemini"\n  draft_thinking_level: "max"\n').draft_thinking_level
    assert _config(BASE + 'generator:\n  engine: "openai"\n').draft_thinking_level is None


def test_polish_temperature():
    assert _config(BASE + 'generator:\n  polish_temperature: 0.1\n').polish_temperature == 0.1
    assert _config(BASE + 'generator:\n  polish_temperature: 0\n').polish_temperature == 0.0
    with pytest.raises(ValueError, match="'polish_temperature' under 'generator'"):
        _config(BASE + 'generator:\n  engine: "gemini"\n').polish_temperature
    for bad in ('"0.1"', 'true'):
        with pytest.raises(ValueError, match="'generator.polish_temperature' must be a number"):
            _config(BASE + f'generator:\n  polish_temperature: {bad}\n').polish_temperature


PRICING = """gemini_pricing:
  pro:
    input_per_million: 2
    output_per_million: 12.0
    long_context:
      threshold_tokens: 200000
      input_per_million: 4.0
      output_per_million: 18
  lite:
    input_per_million: 0.25
    output_per_million: 1.5
"""


def test_gemini_price_valid():
    cfg = _config(BASE + PRICING)
    assert cfg.gemini_price("pro") == {"input_per_million": 2.0, "output_per_million": 12.0, "long_context": {
        "threshold_tokens": 200000, "input_per_million": 4.0, "output_per_million": 18.0}}
    assert cfg.gemini_price("lite") == {"input_per_million": 0.25, "output_per_million": 1.5}


@pytest.mark.parametrize("yaml_text, match", [
    ("", "section 'gemini_pricing'"),
    ("gemini_pricing:\n  other: {input_per_million: 1, output_per_million: 1}\n", "No price for Gemini model 'pro'"),
    ("gemini_pricing:\n  pro: {input_per_million: -1, output_per_million: 1}\n", "input_per_million' must be a non-negative"),
    ("gemini_pricing:\n  pro: {input_per_million: 1}\n", "output_per_million' must be a non-negative"),
    ("gemini_pricing:\n  pro: {input_per_million: true, output_per_million: 1}\n", "non-negative number"),
    ("gemini_pricing:\n  pro: {input_per_million: 1, output_per_million: 1, batch: 1}\n", "Unknown key"),
    ("gemini_pricing:\n  pro: {input_per_million: 1, output_per_million: 1, long_context: {input_per_million: 1, output_per_million: 1}}\n",
     "threshold_tokens' must be a positive integer"),
    ("gemini_pricing:\n  pro: {input_per_million: 1, output_per_million: 1, long_context: {threshold_tokens: 5, input_per_million: 1}}\n",
     "long_context.output_per_million"),
])
def test_gemini_price_invalid(yaml_text, match):
    with pytest.raises(ValueError, match=match):
        _config(BASE + yaml_text).gemini_price("pro")


def test_repo_config_prices_every_configured_gemini_model():
    from pipeline.config import PipelineConfig
    with patch("os.makedirs"):
        cfg = PipelineConfig("config.yaml")
    assert cfg.draft_thinking_level == "high"
    assert cfg.polish_temperature == 0.1
    assert cfg.gemini_price(cfg.gemini_model_name)["output_per_million"] == 12.0
    assert cfg.gemini_price(cfg.pronunciation_model_name)
    assert cfg.gemini_price("gemini-3.1-flash-lite")  # enhance_transcript.py default --model
