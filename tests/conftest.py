import sys
from unittest.mock import MagicMock

# Stub elevenlabs if it is not fully installed (e.g., due to Windows Long Path limits)
try:
    import elevenlabs
    from elevenlabs.client import ElevenLabs
except Exception:
    mock_elevenlabs = MagicMock()
    mock_client = MagicMock()
    
    # Class for VoiceSettings so its attributes are properly populated and checked
    class MockVoiceSettings:
        def __init__(self, stability, similarity_boost):
            self.stability = stability
            self.similarity_boost = similarity_boost

    mock_elevenlabs.VoiceSettings = MockVoiceSettings
    
    # Setup client module stub
    sys.modules["elevenlabs"] = mock_elevenlabs
    sys.modules["elevenlabs.client"] = mock_client
