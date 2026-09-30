import os
import re
import tempfile
from abc import ABC, abstractmethod
from pipeline.logger import get_logger
from openai import OpenAI

logger = get_logger(__name__)


class BaseTTS(ABC):
    """
    Abstract Base Class for Text-to-Speech synthesis engines.  Subclasses that produce MP3 per chunk must implement:
      - _chunk_size (property): max characters per chunk for this engine
      - _synthesize_chunk(text): returns raw MP3 bytes for a single chunk
      - _setup(): hook called once before chunking begins, for lazy imports and client init

    Subclasses with a fundamentally different encoding pipeline (e.g. GeminiTTS,
    which accumulates PCM and encodes at the end) should override synthesize() directly.
    """

    @property
    def file_extension(self) -> str:
        return "mp3"

    @property
    @abstractmethod
    def _chunk_size(self) -> int:
        """Maximum characters per text chunk for this engine."""
        pass

    @abstractmethod
    def _synthesize_chunk(self, text: str) -> bytes:
        """
        Synthesize a single text chunk and return raw MP3 bytes.
        Called by the base synthesize() orchestration loop.
        """
        pass

    def _setup(self) -> None:
        """
        Called once at the start of synthesize() before chunking begins.
        Subclasses use this for lazy imports and client initialization.
        """
        pass

    def synthesize(self, text: str, output_path: str) -> None:
        """
        Template Method: chunks text, synthesizes each chunk, and writes the final file.

        For a single chunk the bytes are written directly.
        For multiple chunks each chunk is written to a temp file, then merged via
        pydub to avoid truncated-tail artifacts from raw MP3 byte concatenation.
        """
        self._setup()
        text_chunks = self._chunk_text(text, max_chars=self._chunk_size)
        logger.info(
            f"Text length {len(text)} split into {len(text_chunks)} chunk(s) for synthesis."
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if len(text_chunks) == 1:
            audio_bytes = self._synthesize_chunk(text_chunks[0])
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
        else:
            with tempfile.TemporaryDirectory() as tmp_dir:
                chunk_paths: list[str] = []
                for idx, chunk in enumerate(text_chunks, 1):
                    logger.info(
                        f"Synthesizing chunk {idx}/{len(text_chunks)} ({len(chunk)} chars)..."
                    )
                    audio_bytes = self._synthesize_chunk(chunk)
                    chunk_path = os.path.join(tmp_dir, f"chunk_{idx:04d}.mp3")
                    with open(chunk_path, "wb") as f:
                        f.write(audio_bytes)
                    chunk_paths.append(chunk_path)

                self._merge_audio_chunks(chunk_paths, output_path)

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

    def _merge_audio_chunks(self, chunk_paths: list[str], output_path: str) -> None:
        """
        Merges a list of MP3 chunk files into a single MP3 using ffmpeg's concat protocol.
        This avoids the truncated-tail bug caused by raw byte concatenation, where
        each MP3 encoder holds back the last few frames in its internal buffer and
        never flushes them when bytes are appended directly.

        Uses imageio-ffmpeg's bundled binary so no system-wide ffmpeg install is needed.
        Bypasses pydub entirely to avoid ffprobe compatibility issues on Windows.
        """
        try:
            import imageio_ffmpeg
            import subprocess
            import tempfile
        except ImportError:
            logger.error("imageio-ffmpeg is required. Install with: pip install imageio-ffmpeg")
            raise

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # Write a concat demuxer input file listing all chunks
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            concat_file = f.name
            for path in chunk_paths:
                # Use absolute paths and escape for ffmpeg's concat demuxer format
                abs_path = os.path.abspath(path)
                escaped = abs_path.replace("\\", "/").replace("'", r"'\''")
                f.write(f"file '{escaped}'\n")

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            subprocess.run(
                [
                    ffmpeg_exe, '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c', 'copy',  # Stream copy — no re-encoding
                    output_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"Merged {len(chunk_paths)} audio chunk(s) into {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg concat failed: {e.stderr}")
            raise
        finally:
            os.unlink(concat_file)


class ElevenLabsTTS(BaseTTS):
    """
    Synthesizes speech using the ElevenLabs API.
    Supports automatic chunking for texts exceeding ElevenLabs' character limit.
    """

    # ElevenLabs streaming endpoint terminates early above ~2000 chars.
    # 1500 is a safe ceiling to account for Unicode normalization variance.
    @property
    def _chunk_size(self) -> int:
        return 1500

    def __init__(
        self,
        api_key: str = None,
        voice_id: str = None,
        model_id: str = None,
        stability: float = None,
        similarity_boost: float = None,
        ssl_verify: bool = True,
    ):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.warning(
                "ElevenLabs API key is not configured. ElevenLabs TTS calls will fail unless keys are provided."
            )

        self.voice_id = voice_id
        self.model_id = model_id
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.ssl_verify = ssl_verify

    def _setup(self) -> None:
        logger.info(
            f"Synthesizing audio via ElevenLabs TTS (Voice: {self.voice_id}, Model: {self.model_id})..."
        )
        try:
            # Import here so missing package gives a clear error at call time
            from elevenlabs.client import ElevenLabs
            from elevenlabs import VoiceSettings
            import httpx

            custom_httpx = httpx.Client(verify=self.ssl_verify, timeout=300.0)
            self._client = ElevenLabs(
                api_key=self.api_key, httpx_client=custom_httpx, timeout=300.0
            )
            self._voice_settings = VoiceSettings(
                stability=self.stability, similarity_boost=self.similarity_boost
            )

        except ImportError:
            logger.error("The 'elevenlabs' package is not installed. Please install it using pip.")
            raise
        except Exception as e:
            logger.error(f"ElevenLabs TTS synthesis failed: {e}")
            raise

    def _synthesize_chunk(self, text: str) -> bytes:
        audio_generator = self._client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_128",
            optimize_streaming_latency=None,
            voice_settings=self._voice_settings,
        )
        return b"".join(audio_generator)


class GoogleCloudTTS(BaseTTS):
    """
    Synthesizes speech using Google Cloud Text-to-Speech API.
    """

    # 2000 chars is safe for Hebrew (2-byte chars) within Google's 5000-byte limit.
    @property
    def _chunk_size(self) -> int:
        return 2000

    def __init__(
        self,
        credentials_path: str = None,
        voice_name: str = "he-IL-Neural2-M",
        language_code: str = "he-IL",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ):
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.warning(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable is not set. Google TTS calls may fail."
            )

        self.voice_name = voice_name
        self.language_code = language_code
        self.speaking_rate = speaking_rate
        self.pitch = pitch

    def _setup(self) -> None:
        logger.info(
            f"Synthesizing audio via Google Cloud TTS (Voice: {self.voice_name}, Rate: {self.speaking_rate})..."
        )
        try:
            from google.cloud import texttospeech

            client = texttospeech.TextToSpeechClient()
            self._google_client = client
            self._voice = texttospeech.VoiceSelectionParams(
                language_code=self.language_code, name=self.voice_name
            )
            self._audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch,
            )
            self._texttospeech = texttospeech

        except ImportError:
            logger.error(
                "The 'google-cloud-texttospeech' package is not installed. Please install it using pip."
            )
            raise
        except Exception as e:
            logger.error(f"Google Cloud TTS synthesis failed: {e}")
            raise

    def _synthesize_chunk(self, text: str) -> bytes:
        synthesis_input = self._texttospeech.SynthesisInput(text=text)
        response = self._google_client.synthesize_speech(
            input=synthesis_input, voice=self._voice, audio_config=self._audio_config
        )
        return response.audio_content


class OpenAITTS(BaseTTS):
    """
    Synthesizes speech using the OpenAI Text-to-Speech API.
    Supports automatic chunking for texts exceeding OpenAI's character limit.
    """

    # OpenAI's hard limit is 4096 chars; 4000 gives a safe buffer.
    @property
    def _chunk_size(self) -> int:
        return 4000

    def __init__(
        self,
        api_key: str = None,
        voice: str = "alloy",
        model: str = "tts-1",
        speed: float = 1.0,
        ssl_verify: bool = True,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning(
                "OpenAI API key is not configured. OpenAI TTS calls will fail unless keys are provided."
            )

        self.voice = voice
        self.model = model
        self.speed = speed
        self.ssl_verify = ssl_verify

    def _setup(self) -> None:
        logger.info(
            f"Synthesizing audio via OpenAI TTS (Voice: {self.voice}, Model: {self.model}, Speed: {self.speed})..."
        )
        try:
            import httpx

            custom_httpx = httpx.Client(verify=self.ssl_verify, timeout=300.0)
            self._client = OpenAI(api_key=self.api_key, http_client=custom_httpx)

        except ImportError:
            logger.error("The 'openai' package is not installed. Please install it using pip.")
            raise
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis failed: {e}")
            raise

    def _synthesize_chunk(self, text: str) -> bytes:
        response = self._client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
            speed=self.speed,
        )
        return response.content


class GeminiTTS(BaseTTS):
    """
    Synthesizes speech using the Gemini 3.1 TTS preview models via the GCP Text-to-Speech API.
    Bypasses the buggy google-genai SDK and bills based on input characters instead of output tokens.

    This engine accumulates raw LINEAR16 PCM from all chunks, then encodes the combined
    stream to MP3 in a single lameenc pass — which avoids inter-chunk discontinuities that
    would occur if each chunk were encoded independently and concatenated.  Because of this
    PCM-accumulate-then-encode flow the base class synthesize() loop is not applicable here,
    so GeminiTTS overrides synthesize() directly.
    """

    # Gemini TTS uses max_chars=2000 to stay within the API's length limits.
    @property
    def _chunk_size(self) -> int:
        return 2000

    def __init__(
        self,
        model_id: str,
        voice_name: str,
        api_key: str = None,
    ):
        self.model_id = model_id
        self.voice_name = voice_name

        try:
            import google.auth
            self.credentials, self.project = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        except Exception as e:
            logger.warning(
                f"Failed to load Google Cloud credentials: {e}. Gemini TTS calls may fail."
            )
            self.credentials = None
            self.project = None

    @property
    def file_extension(self) -> str:
        return "mp3"

    def _synthesize_chunk(self, text: str) -> bytes:
        """Not used — GeminiTTS overrides synthesize() to do PCM accumulation."""
        raise NotImplementedError("GeminiTTS uses a custom synthesize() flow.")

    def synthesize(self, text: str, output_path: str) -> None:
        logger.info(
            f"Synthesizing audio via Gemini GCP TTS (Model: {self.model_id}, Voice: {self.voice_name})..."
        )
        try:
            import requests
            import base64
            import lameenc
            from google.auth.transport.requests import Request

            if not self.credentials:
                raise ValueError(
                    "GCP Credentials not loaded. Ensure GOOGLE_APPLICATION_CREDENTIALS is set."
                )

            self.credentials.refresh(Request())
            token = self.credentials.token

            text_chunks = self._chunk_text(text, max_chars=self._chunk_size)
            logger.info(
                f"Text length {len(text)} split into {len(text_chunks)} chunk(s) for Gemini TTS."
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            url = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
                "x-goog-user-project": self.project if self.project else "",
            }

            # Accumulate all PCM before encoding so lameenc can produce a single
            # gapless MP3 stream rather than spliced per-chunk files.
            all_pcm_data = b""
            for idx, chunk in enumerate(text_chunks, 1):
                logger.info(
                    f"Generating audio for chunk {idx}/{len(text_chunks)} ({len(chunk)} chars)..."
                )

                payload = {
                    "audioConfig": {
                        "audioEncoding": "LINEAR16",
                        "pitch": 0,
                        "speakingRate": 1,
                    },
                    "input": {"text": chunk},
                    "voice": {
                        "languageCode": "he-il",
                        "modelName": self.model_id,
                        "name": self.voice_name,
                    },
                }

                if hasattr(self, "prompt") and self.prompt:
                    payload["input"]["prompt"] = self.prompt

                response = requests.post(url, headers=headers, json=payload, timeout=120)

                if response.status_code != 200:
                    logger.error(f"GCP API Error: {response.text}")
                    raise ValueError(
                        f"GCP TTS API failed with status {response.status_code}"
                    )

                data = response.json()
                audio_content = data.get("audioContent")
                if not audio_content:
                    raise ValueError("No audioContent returned by GCP API.")

                all_pcm_data += base64.b64decode(audio_content)

            logger.info("Encoding raw PCM to MP3 using lameenc...")
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(24000)
            encoder.set_channels(1)
            encoder.set_quality(2)

            mp3_data = encoder.encode(all_pcm_data)
            mp3_data += encoder.flush()

            logger.info(f"Writing compressed MP3 to {output_path}...")
            with open(output_path, "wb") as f:
                f.write(mp3_data)

            logger.info(
                f"Successfully saved compressed Gemini native TTS audio to {output_path}"
            )

        except Exception as e:
            logger.error(f"Gemini native TTS synthesis failed: {e}")
            raise
