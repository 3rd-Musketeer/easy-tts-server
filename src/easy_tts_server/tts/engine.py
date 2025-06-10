import logging
from collections.abc import Generator

import numpy as np

from ..segmenter import TextSegmenter
from .streaming_buffer import StreamingBuffer
from .synthesis_core import SynthesisCore
from .task_manager import TaskManager

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    A TTS engine that supports both English and Chinese using Kokoro models.
    Features streaming synthesis with background threading for reduced latency.
    Language must be explicitly specified by the user.
    """

    def __init__(self, repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh"):
        self.synthesis_core = SynthesisCore(repo_id)
        self.segmenter = TextSegmenter()
        self.task_manager = TaskManager(self.synthesis_core)
        self.streaming_buffer = StreamingBuffer(self.segmenter, self.task_manager)
        logger.info("TTSEngine initialized")

    def tts(self, text: str, language: str, voice: str | None = None) -> np.ndarray:
        """
        Synthesize entire text to audio.

        Args:
            text: Input text to synthesize
            language: Language code ('en' or 'zh'). Must be provided.
            voice: Voice name. If None, uses default voice for the language.

        Returns:
            Audio as numpy array
        """
        if not text.strip():
            return np.array([], dtype=np.float32)

        # Validate language
        if language not in self.synthesis_core.get_available_languages():
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {self.synthesis_core.get_available_languages()}"
            )

        voice = voice or self.synthesis_core.get_default_voice(language)

        segments = self.segmenter.segment_sentences(text, language=language)
        audio_segments = [
            self.synthesis_core.synthesize(seg, language, voice) for seg in segments
        ]

        return (
            np.concatenate(audio_segments)
            if audio_segments
            else np.array([], dtype=np.float32)
        )

    def tts_stream(
        self, text: str, language: str, voice: str | None = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream audio synthesis with background threading.

        Args:
            text: Input text to synthesize
            language: Language code ('en' or 'zh'). Must be provided.
            voice: Voice name. If None, uses default voice for the language.

        Yields:
            Audio chunks as numpy arrays
        """
        if not text.strip():
            return

        # Validate language
        if language not in self.synthesis_core.get_available_languages():
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {self.synthesis_core.get_available_languages()}"
            )

        # Normalize text the same way streaming buffer does
        normalized_text = " ".join(text.split())

        voice = voice or self.synthesis_core.get_default_voice(language)
        segments = self.segmenter.segment_sentences(normalized_text, language=language)

        # Set batch mode (not streaming)
        self.task_manager.set_streaming_mode(False)
        self.task_manager.queue_tasks(segments, language, voice)
        yield from self.task_manager.get_audio()

    def feed_in(self, token: str, language: str) -> None:
        """
        Feed tokens from streaming LLM input.

        Args:
            token: Text token to process
            language: Language code ('en' or 'zh'). Must be provided.
        """
        # Validate language
        if language not in self.synthesis_core.get_available_languages():
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {self.synthesis_core.get_available_languages()}"
            )

        # Set streaming mode on first feed_in
        self.task_manager.set_streaming_mode(True)
        self.streaming_buffer.feed_token(token, language)

    def feed_out(self) -> Generator[np.ndarray, None, None]:
        """Stream audio segments as they become available."""
        yield from self.task_manager.get_audio()

    def flush(self) -> None:
        """Force process any remaining content in buffer."""
        self.streaming_buffer.flush()
        # Signal that streaming input is complete
        self.task_manager.signal_streaming_complete()

    def reset(self) -> None:
        """Reset all TTS operations and cancel ongoing synthesis."""
        self.task_manager.reset()
        self.streaming_buffer.reset()

    def get_available_languages(self) -> list[str]:
        return self.synthesis_core.get_available_languages()

    def get_voices(self, language: str) -> list[str]:
        return self.synthesis_core.get_voices(language)

    @property
    def sample_rate(self) -> int:
        """Get the sample rate from synthesis core."""
        return self.synthesis_core.sample_rate


def create_tts_engine(repo_id: str = "hexgrad/Kokoro-82M-v1.1-zh") -> TTSEngine:
    """Create and return a TTS engine instance."""
    return TTSEngine(repo_id=repo_id)
