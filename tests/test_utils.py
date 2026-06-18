import os
import pytest
from unittest.mock import patch, MagicMock
from pipeline.utils import save_output_file

def test_save_output_file_text_success(tmp_path):
    # Setup directories
    output_dir = tmp_path / "output"
    
    logger = MagicMock()
    content = "test transcript content"
    
    # Save the file
    history_path = save_output_file(
        directory=str(output_dir),
        base_name="Yoreh_Deah_Siman_94_gpt-5_draft",
        extension="txt",
        content=content,
        logger=logger
    )
    
    # Verify both copies exist
    assert os.path.exists(history_path)
    latest_path = os.path.join(str(output_dir), "Yoreh_Deah_Siman_94_gpt-5_draft.txt")
    assert os.path.exists(latest_path)
    
    # Verify content
    with open(latest_path, "r", encoding="utf-8") as f:
        assert f.read() == content
        
    with open(history_path, "r", encoding="utf-8") as f:
        assert f.read() == content
        
    logger.info.assert_any_call(f"Saved history copy: {history_path}")
    logger.info.assert_any_call(f"Updated latest copy at: {latest_path}")

def test_save_output_file_callback_success(tmp_path):
    output_dir = tmp_path / "output"
    logger = MagicMock()
    
    # Mock synthesis callback
    mock_callback = MagicMock()
    
    history_path = save_output_file(
        directory=str(output_dir),
        base_name="Yoreh_Deah_Siman_94_gpt-5",
        extension="mp3",
        content=None,
        logger=logger,
        write_callback=mock_callback
    )
    
    # Verify callback was called with the history path
    mock_callback.assert_called_once_with(history_path)
    
    # Since mock_callback doesn't actually write a file, shutil.copyfile would fail in real life.
    # But wait! If the callback doesn't create the file, shutil.copyfile(history_path, latest_path) will raise FileNotFoundError.
    # In our test, let's verify that when history_path is created, copy works.
    
def test_save_output_file_callback_with_actual_file(tmp_path):
    output_dir = tmp_path / "output"
    logger = MagicMock()
    
    def dummy_writer(path):
        with open(path, "wb") as f:
            f.write(b"fake mp3 audio bytes")
            
    history_path = save_output_file(
        directory=str(output_dir),
        base_name="Yoreh_Deah_Siman_94_gpt-5",
        extension="mp3",
        content=None,
        logger=logger,
        write_callback=dummy_writer
    )
    
    assert os.path.exists(history_path)
    latest_path = os.path.join(str(output_dir), "Yoreh_Deah_Siman_94_gpt-5.mp3")
    assert os.path.exists(latest_path)
    
    with open(latest_path, "rb") as f:
        assert f.read() == b"fake mp3 audio bytes"

@patch("shutil.copyfile")
def test_save_output_file_lock_handles_gracefully(mock_copyfile, tmp_path):
    output_dir = tmp_path / "output"
    logger = MagicMock()
    
    # Simulate a file permission/lock error on copyfile
    mock_copyfile.side_effect = PermissionError("[Errno 13] Permission denied")
    
    content = "locked file test"
    history_path = save_output_file(
        directory=str(output_dir),
        base_name="Yoreh_Deah_Siman_94_gpt-5_draft",
        extension="txt",
        content=content,
        logger=logger
    )
    
    # History copy should still be written (since open(history_path) doesn't use copyfile)
    assert os.path.exists(history_path)
    with open(history_path, "r", encoding="utf-8") as f:
        assert f.read() == content
        
    # Latest path copy should have failed, and logged a warning
    latest_path = os.path.join(str(output_dir), "Yoreh_Deah_Siman_94_gpt-5_draft.txt")
    logger.warning.assert_called_once()
    assert "Could not overwrite latest copy at" in logger.warning.call_args[0][0]
    
    # Function should still return history_path successfully
    assert history_path is not None
