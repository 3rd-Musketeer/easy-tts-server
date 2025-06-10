import logging

import numpy as np
import torch
from kokoro import KModel, KPipeline

logger = logging.getLogger(__name__)


class SynthesisCore:
    """Pure model operations - no threading, just synthesis."""

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 24000
        self.speed = 1.1
        self._load_models()

        # Available voices and defaults
        self.voices = {
            "en": ["af_maple", "af_sol", "bf_vale"],
            "zh": ["zf_001", "zm_010"],
        }
        self.default_voices = {"en": "af_maple", "zh": "zf_001"}

        self._warmup_models()

    def _warmup_models(self):
        """
        Run a dummy synthesis to warm up the models and avoid a 'cold start'
        on the first real request, which can cause noise or latency.
        """
        logger.info("Warming up TTS models...")
        try:
            # Warm up English model
            self.synthesize("Hello.", "en", self.default_voices["en"])
            logger.info("English TTS model is warm.")

            # Warm up Chinese model
            self.synthesize("你好。", "zh", self.default_voices["zh"])
            logger.info("Chinese TTS model is warm.")

            logger.info("TTS models warmed up successfully.")
        except Exception as e:
            logger.error(f"An error occurred during model warm-up: {e}")

    def _load_models(self):
        """Load Kokoro model and create pipelines."""
        self.model = KModel(repo_id=self.repo_id).to(self.device).eval()

        # Create English pipeline for handling English words in Chinese text
        self.en_pipeline = KPipeline(lang_code="a", repo_id=self.repo_id, model=False)

        self.pipelines = {
            "en": {
                "american": KPipeline(
                    lang_code="a", repo_id=self.repo_id, model=self.model
                ),
                "british": KPipeline(
                    lang_code="b", repo_id=self.repo_id, model=self.model
                ),
            },
            "zh": KPipeline(
                lang_code="z",
                repo_id=self.repo_id,
                model=self.model,
                en_callable=self._en_callable,
            ),
        }

    def _en_callable(self, text):
        """Handle English words mixed in Chinese text."""
        return next(self.en_pipeline(text)).phonemes

    def _clean_audio(self, audio) -> np.ndarray:
        """Clean up TTS audio artifacts like noise at beginning/end."""
        if audio is None:
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        # Convert torch tensor to numpy array if needed
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()

        # Ensure it's a numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)

        if len(audio) == 0:
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Remove DC offset
        audio = audio - np.mean(audio)

        # Apply gentle fade-in to remove initial artifacts (first 100ms)
        fade_samples = min(int(0.1 * self.sample_rate), len(audio) // 4)
        if fade_samples > 0:
            fade_in = np.linspace(0, 1, fade_samples)
            audio[:fade_samples] *= fade_in

        # Apply gentle fade-out to remove trailing artifacts (last 50ms)
        fade_out_samples = min(int(0.05 * self.sample_rate), len(audio) // 8)
        if fade_out_samples > 0:
            fade_out = np.linspace(1, 0, fade_out_samples)
            audio[-int(fade_out_samples) :] *= fade_out

        # Trim silence from beginning and end
        audio = self._trim_silence(audio)

        # Normalize to prevent clipping while preserving dynamics
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude > 0.95:
            audio = audio * (0.95 / max_amplitude)

        return audio

    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        if len(audio) == 0:
            return audio

        # Find first and last non-silent samples
        abs_audio = np.abs(audio)
        non_silent = abs_audio > threshold

        if not np.any(non_silent):
            # If entire audio is silent, return a small portion
            return audio[: int(0.1 * self.sample_rate)]

        first_sound = int(np.argmax(non_silent))
        last_sound = int(len(audio) - np.argmax(non_silent[::-1]) - 1)

        # Keep a small padding around the sound
        padding = int(0.02 * self.sample_rate)  # 20ms padding
        start = max(0, first_sound - padding)
        end = min(len(audio), last_sound + padding)

        return audio[start:end]

    def synthesize(self, text: str, language: str, voice: str) -> np.ndarray:
        """Synthesize text to audio."""
        # Skip empty or whitespace-only text
        if not text or not text.strip():
            return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

        # Fallback to 'en' if language is None or invalid
        if not language or language not in self.voices:
            language = "en"
            logger.warning("Invalid language, falling back to English")

        # Use default voice if voice is invalid
        if not voice or voice not in self.voices.get(language, []):
            voice = self.get_default_voice(language)

        try:
            pipeline = self._get_pipeline(language, voice)
            generator = pipeline(text, voice=voice, speed=self.speed)
            result = next(generator)

            # Clean up audio artifacts
            audio = result.audio
            audio = self._clean_audio(audio)
            return audio
        except Exception as e:
            logger.error(f"Synthesis failed for '{text[:30]}...': {e}")
            return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)

    def _get_pipeline(self, language: str, voice: str):
        """Get appropriate pipeline for language and voice."""
        if language == "en":
            return self.pipelines["en"]["british" if voice == "bf_vale" else "american"]
        elif language == "zh":
            return self.pipelines["zh"]
        else:
            raise ValueError(f"Unsupported language: {language}")

    def get_available_languages(self) -> list[str]:
        return list(self.voices.keys())

    def get_voices(self, language: str) -> list[str]:
        return self.voices.get(language, [])

    def get_default_voice(self, language: str) -> str:
        return self.default_voices.get(language, self.default_voices["en"])
