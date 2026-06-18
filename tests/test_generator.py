import pytest
import os
from unittest.mock import patch, MagicMock
from pipeline.generator import GeminiScriptGenerator, OpenAIScriptGenerator

@patch("google.genai.Client")
def test_generator_initialization(mock_client_cls):
    # Tests that client is initialized with correct credentials
    generator = GeminiScriptGenerator(api_key="fake-key", model_name="fake-model", temperature=0.5)
    mock_client_cls.assert_called_once_with(api_key="fake-key")
    assert generator.model_name == "fake-model"
    assert generator.temperature == 0.5

@patch("google.genai.Client")
def test_generate_siman_script_success(mock_client_cls):
    # Setup mock client and mock response
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = "שיעור שמע לדוגמה"
    mock_client.models.generate_content.return_value = mock_response

    generator = GeminiScriptGenerator(api_key="fake-key")
    script = generator.generate_siman_script(
        siman=94,
        master_context="קובץ מקורות כלשהו",
        system_instruction="הנחיות מערכת"
    )

    assert script == "שיעור שמע לדוגמה"
    # Verify client call structure
    mock_client.models.generate_content.assert_called_once()
    call_args, call_kwargs = mock_client.models.generate_content.call_args
    assert call_kwargs["model"] == "gemini-3.5-flash"
    assert "צ\"ד" in call_kwargs["contents"] # Verifies Siman 94 gematria is in prompt
    assert call_kwargs["config"].temperature == 0.3
    assert call_kwargs["config"].system_instruction == "הנחיות מערכת"


@patch("google.genai.Client")
def test_polish_siman_script_success(mock_client_cls):
    # Setup mock client and mock response
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = "שיעור שמע מוגה ומלוטש"
    mock_client.models.generate_content.return_value = mock_response

    generator = GeminiScriptGenerator(api_key="fake-key", model_name="fake-model")
    polished = generator.polish_siman_script("טקסט גולמי", "", "עורך לשוני מקצועי")

    assert polished == "שיעור שמע מוגה ומלוטש"
    mock_client.models.generate_content.assert_called_once()
    call_args, call_kwargs = mock_client.models.generate_content.call_args
    assert call_kwargs["model"] == "fake-model"
    assert call_kwargs["contents"] == "טקסט גולמי"
    assert call_kwargs["config"].temperature == 0.1
    assert call_kwargs["config"].system_instruction.startswith("עורך לשוני מקצועי")



@patch("pipeline.generator.OpenAI")
def test_openai_generator_initialization(mock_openai_cls):
    generator = OpenAIScriptGenerator(api_key="fake-openai-key", model_name="gpt-4o", temperature=0.7)
    mock_openai_cls.assert_called_once_with(api_key="fake-openai-key")
    assert generator.model_name == "gpt-4o"
    assert generator.temperature == 0.7
    assert generator.service_tier is None

    generator_with_tier = OpenAIScriptGenerator(api_key="fake-openai-key", model_name="o1", service_tier="flex")
    assert generator_with_tier.service_tier == "flex"


@patch("pipeline.generator.OpenAI")
def test_openai_generate_siman_script_gpt4o(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "שיעור שמע מאופנאיי"
    mock_client.chat.completions.create.return_value = mock_response

    generator = OpenAIScriptGenerator(api_key="fake-key", model_name="gpt-4o", temperature=0.5)
    script = generator.generate_siman_script(
        siman=94,
        master_context="קובץ מקורות כלשהו",
        system_instruction="הנחיות מערכת"
    )

    assert script == "שיעור שמע מאופנאיי"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["messages"][0] == {"role": "system", "content": "הנחיות מערכת"}


@patch("pipeline.generator.OpenAI")
def test_openai_generate_siman_script_with_service_tier(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "שיעור שמע מאופנאיי עם שירות פלקס"
    mock_client.chat.completions.create.return_value = mock_response

    generator = OpenAIScriptGenerator(api_key="fake-key", model_name="gpt-4o", temperature=0.5, service_tier="flex")
    script = generator.generate_siman_script(
        siman=94,
        master_context="קובץ מקורות כלשהו",
        system_instruction="הנחיות מערכת"
    )

    assert script == "שיעור שמע מאופנאיי עם שירות פלקס"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["service_tier"] == "flex"


@patch("pipeline.generator.OpenAI")
def test_openai_generate_siman_script_o1_reasoning(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "שיעור שמע מודל o1"
    mock_client.chat.completions.create.return_value = mock_response

    # Initialize with o1 reasoning model
    generator = OpenAIScriptGenerator(api_key="fake-key", model_name="o1", temperature=1.0)
    script = generator.generate_siman_script(
        siman=94,
        master_context="קובץ מקורות כלשהו",
        system_instruction="הנחיות מערכת"
    )

    assert script == "שיעור שמע מודל o1"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "o1"
    # Verify temperature is not passed for o1 reasoning model
    assert "temperature" not in call_kwargs
    # Verify developer message is used
    assert call_kwargs["messages"][0] == {"role": "developer", "content": "הנחיות מערכת"}


@patch("pipeline.generator.OpenAI")
def test_openai_polish_siman_script_success(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "שיעור שמע מלוטש מאופנאיי"
    mock_client.chat.completions.create.return_value = mock_response

    generator = OpenAIScriptGenerator(api_key="fake-key", model_name="gpt-4o")
    polished = generator.polish_siman_script("טקסט גולמי", "", "עורך לשוני מקצועי")

    assert polished == "שיעור שמע מלוטש מאופנאיי"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][0]["content"].startswith("עורך לשוני מקצועי")


@patch("pipeline.generator.OpenAI")
def test_openai_polish_siman_script_with_service_tier(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "שיעור שמע מלוטש מאופנאיי עם שירות פלקס"
    mock_client.chat.completions.create.return_value = mock_response

    generator = OpenAIScriptGenerator(api_key="fake-key", model_name="gpt-4o", service_tier="flex")
    polished = generator.polish_siman_script("טקסט גולמי", "", "עורך לשוני מקצועי")

    assert polished == "שיעור שמע מלוטש מאופנאיי עם שירות פלקס"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["service_tier"] == "flex"


@patch("pipeline.generator.OpenAI")
def test_openai_create_batch_generation_job(mock_openai_cls, tmp_path):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_file_response = MagicMock()
    mock_file_response.id = "file-123"
    mock_client.files.create.return_value = mock_file_response
    
    mock_batch_response = MagicMock()
    mock_batch_response.id = "batch-555"
    mock_client.batches.create.return_value = mock_batch_response

    generator = OpenAIScriptGenerator(api_key="fake-key", model_name="o1")
    batch_file = os.path.join(tmp_path, "batch_input.jsonl")
    
    batch_id = generator.create_batch_generation_job(
        siman=94,
        master_context="קובץ מקורות",
        system_instruction="הוראות",
        batch_input_path=batch_file
    )

    assert batch_id == "batch-555"
    assert os.path.exists(batch_file)
    with open(batch_file, "r", encoding="utf-8") as f:
        line = f.readline()
        import json
        payload = json.loads(line)
        assert payload["custom_id"] == "siman_94_generation"
        assert payload["body"]["model"] == "o1"
        assert "temperature" not in payload["body"]

    mock_client.files.create.assert_called_once()
    mock_client.batches.create.assert_called_once_with(
        input_file_id="file-123",
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )


@patch("pipeline.generator.OpenAI")
def test_openai_retrieve_batch_result_completed(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_batch = MagicMock()
    mock_batch.status = "completed"
    mock_batch.output_file_id = "output-file-456"
    mock_client.batches.retrieve.return_value = mock_batch
    
    mock_content_response = MagicMock()
    mock_content_response.text = '{"custom_id": "siman_94_generation", "response": {"status_code": 200, "body": {"choices": [{"message": {"content": "שיעור הלכה שלם ומפורט"}}]}}}\n'
    mock_client.files.content.return_value = mock_content_response

    generator = OpenAIScriptGenerator(api_key="fake-key", model_name="o1")
    result = generator.retrieve_batch_result("batch-555")

    assert result["status"] == "completed"
    assert result["content"] == "שיעור הלכה שלם ומפורט"
    mock_client.batches.retrieve.assert_called_once_with("batch-555")
    mock_client.files.content.assert_called_once_with("output-file-456")


@patch("pipeline.generator.OpenAI")
def test_openai_retrieve_batch_result_running(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_batch = MagicMock()
    mock_batch.status = "in_progress"
    mock_client.batches.retrieve.return_value = mock_batch

    generator = OpenAIScriptGenerator(api_key="fake-key", model_name="o1")
    result = generator.retrieve_batch_result("batch-555")

    assert result["status"] == "in_progress"
    assert "content" not in result


@patch("google.genai.Client")
def test_gemini_analyze_cross_relations_success(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.text = "ניתוח יחסים בין סימנים"
    mock_client.models.generate_content.return_value = mock_response

    generator = GeminiScriptGenerator(api_key="fake-key", model_name="fake-model")
    drafts = {94: "טיוטה 94", 95: "טיוטה 95"}
    relations = generator.analyze_cross_relations(drafts, "הנחיית יחסים")

    assert relations == "ניתוח יחסים בין סימנים"
    mock_client.models.generate_content.assert_called_once()
    call_args, call_kwargs = mock_client.models.generate_content.call_args
    assert call_kwargs["model"] == "fake-model"
    assert "=== סימן 94 ===" in call_kwargs["contents"]
    assert "=== סימן 95 ===" in call_kwargs["contents"]
    assert call_kwargs["config"].system_instruction == "הנחיית יחסים"


@patch("pipeline.generator.OpenAI")
def test_openai_analyze_cross_relations_success(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ניתוח יחסים מאופנאיי"
    mock_client.chat.completions.create.return_value = mock_response

    generator = OpenAIScriptGenerator(api_key="fake-key", model_name="gpt-4o")
    drafts = {94: "טיוטה 94", 95: "טיוטה 95"}
    relations = generator.analyze_cross_relations(drafts, "הנחיית יחסים")

    assert relations == "ניתוח יחסים מאופנאיי"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["messages"][0] == {"role": "system", "content": "הנחיית יחסים"}
    assert "=== סימן 94 ===" in call_kwargs["messages"][1]["content"]
    assert "=== סימן 95 ===" in call_kwargs["messages"][1]["content"]

