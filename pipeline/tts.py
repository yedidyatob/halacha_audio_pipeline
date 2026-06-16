import os
import re
from abc import ABC, abstractmethod
from pipeline.logger import get_logger
from openai import OpenAI

logger = get_logger(__name__)

class BaseTTS(ABC):
    """
    Abstract Base Class for Text-to-Speech synthesis engines.
    """
    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> None:
        """
        Synthesizes the given text to an audio file (.mp3) at output_path.
        """
        pass

    def _chunk_text(self, text: str, max_chars: int = 4000) -> list:
        """
        Splits text into chunks of at most max_chars, splitting on paragraph or sentence boundaries.
        """
        if len(text) <= max_chars:
            return [text]
            
        paragraphs = text.split("\n")
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) > max_chars:
                # Flush existing chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split by punctuation
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if len(sentence) > max_chars:
                        words = sentence.split(" ")
                        for word in words:
                            if current_length + len(word) + 1 > max_chars:
                                chunks.append("\n\n".join(current_chunk))
                                current_chunk = [word]
                                current_length = len(word)
                            else:
                                current_chunk.append(word)
                                current_length += len(word) + 1
                    else:
                        if current_length + len(sentence) + 1 > max_chars:
                            chunks.append("\n\n".join(current_chunk))
                            current_chunk = [sentence]
                            current_length = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_length += len(sentence) + 1
            else:
                if current_length + len(para) + 2 > max_chars:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
                    current_length = len(para)
                else:
                    current_chunk.append(para)
                    current_length += len(para) + 2
                    
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks


class ElevenLabsTTS(BaseTTS):
    """
    Synthesizes speech using the ElevenLabs API.
    Supports automatic chunking for texts exceeding ElevenLabs' character limit.
    """
    def __init__(
        self, 
        api_key: str = None, 
        voice_id: str = "pNInz6obpgDQGcFmaJgB", 
        model_id: str = "eleven_v3",
        stability: float = 0.5,
        similarity_boost: float = 0.75
    ):
        # Fallback to ELEVENLABS_API_KEY environment variable if not provided
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.warning("ElevenLabs API key is not configured. ElevenLabs TTS calls will fail unless keys are provided.")
            
        self.voice_id = voice_id
        self.model_id = model_id
        self.stability = stability
        self.similarity_boost = similarity_boost


    def synthesize(self, text: str, output_path: str) -> None:
        logger.info(f"Synthesizing audio via ElevenLabs TTS (Voice: {self.voice_id}, Model: {self.model_id})...")
        try:
            from elevenlabs.client import ElevenLabs
            from elevenlabs import VoiceSettings
            import httpx
            
            # Using verify=False to bypass certificate errors in environments with SSL inspection
            custom_httpx = httpx.Client(verify=False)
            client = ElevenLabs(api_key=self.api_key, httpx_client=custom_httpx)
            
            # Chunk the text to stay within ElevenLabs' request limits (e.g. 4000 chars to be safe)
            text_chunks = self._chunk_text(text, max_chars=4000)
            logger.info(f"Text length {len(text)} split into {len(text_chunks)} chunk(s) for synthesis.")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # We open the file in write-binary mode and append chunks
            with open(output_path, "wb") as f:
                for idx, chunk in enumerate(text_chunks, 1):
                    logger.info(f"Synthesizing chunk {idx}/{len(text_chunks)} ({len(chunk)} chars)...")
                    audio_generator = client.text_to_speech.convert(
                        text=chunk,
                        voice_id=self.voice_id,
                        model_id=self.model_id,
                        voice_settings=VoiceSettings(
                            stability=self.stability,
                            similarity_boost=self.similarity_boost
                        )
                    )
                    for audio_chunk in audio_generator:
                        f.write(audio_chunk)
                        
            logger.info(f"Successfully saved concatenated ElevenLabs audio to {output_path}")
            
        except ImportError:
            logger.error("The 'elevenlabs' package is not installed. Please install it using pip.")
            raise
        except Exception as e:
            logger.error(f"ElevenLabs TTS synthesis failed: {e}")
            raise


class GoogleCloudTTS(BaseTTS):
    """
    Synthesizes speech using Google Cloud Text-to-Speech API.
    """
    def __init__(
        self, 
        credentials_path: str = None, 
        voice_name: str = "he-IL-Neural2-M",
        language_code: str = "he-IL",
        speaking_rate: float = 1.0,
        pitch: float = 0.0
    ):
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
        # Verify credentials exist
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. Google TTS calls may fail.")
            
        self.voice_name = voice_name
        self.language_code = language_code
        self.speaking_rate = speaking_rate
        self.pitch = pitch

    def synthesize(self, text: str, output_path: str) -> None:
        logger.info(f"Synthesizing audio via Google Cloud TTS (Voice: {self.voice_name}, Rate: {self.speaking_rate})...")
        try:
            from google.cloud import texttospeech
            
            client = texttospeech.TextToSpeechClient()
            
            # Chunk the text to stay within Google's 5000 byte limit (use 2000 chars to be safe for 2-byte Hebrew)
            text_chunks = self._chunk_text(text, max_chars=2000)
            logger.info(f"Text length {len(text)} split into {len(text_chunks)} chunk(s) for Google TTS.")
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=self.language_code,
                name=self.voice_name
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as out:
                for idx, chunk in enumerate(text_chunks, 1):
                    logger.info(f"Synthesizing chunk {idx}/{len(text_chunks)} ({len(chunk)} chars)...")
                    synthesis_input = texttospeech.SynthesisInput(text=chunk)
                    response = client.synthesize_speech(
                        input=synthesis_input, 
                        voice=voice, 
                        audio_config=audio_config
                    )
                    out.write(response.audio_content)
                
            logger.info(f"Successfully saved concatenated Google TTS audio to {output_path}")
            
        except ImportError:
            logger.error("The 'google-cloud-texttospeech' package is not installed. Please install it using pip.")
            raise
        except Exception as e:
            logger.error(f"Google Cloud TTS synthesis failed: {e}")
            raise


class OpenAITTS(BaseTTS):
    """
    Synthesizes speech using the OpenAI Text-to-Speech API.
    Supports automatic chunking for texts exceeding OpenAI's character limit.
    """
    def __init__(
        self, 
        api_key: str = None, 
        voice: str = "alloy", 
        model: str = "tts-1",
        speed: float = 1.0
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key is not configured. OpenAI TTS calls will fail unless keys are provided.")
            
        self.voice = voice
        self.model = model
        self.speed = speed

    def synthesize(self, text: str, output_path: str) -> None:
        logger.info(f"Synthesizing audio via OpenAI TTS (Voice: {self.voice}, Model: {self.model}, Speed: {self.speed})...")
        try:
            import httpx
            
            # Using verify=False to bypass certificate errors in corporate networks
            custom_httpx = httpx.Client(verify=False)
            client = OpenAI(api_key=self.api_key, http_client=custom_httpx)
            
            # Chunk the text to stay within OpenAI's 4096 character limit
            text_chunks = self._chunk_text(text, max_chars=4000)
            logger.info(f"Text length {len(text)} split into {len(text_chunks)} chunk(s) for OpenAI TTS.")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "wb") as f:
                for idx, chunk in enumerate(text_chunks, 1):
                    logger.info(f"Synthesizing chunk {idx}/{len(text_chunks)} ({len(chunk)} chars)...")
                    response = client.audio.speech.create(
                        model=self.model,
                        voice=self.voice,
                        input=chunk,
                        speed=self.speed
                    )
                    f.write(response.content)
                        
            logger.info(f"Successfully saved concatenated OpenAI TTS audio to {output_path}")
            
        except ImportError:
            logger.error("The 'openai' package is not installed. Please install it using pip.")
            raise
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis failed: {e}")
            raise
