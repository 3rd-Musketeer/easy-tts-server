import logging

logger = logging.getLogger(__name__)

# Streaming configuration
DETECTION_INTERVAL = 10


class StreamingBuffer:
    """Handle feed_in logic + dynamic segmentation."""

    def __init__(self, segmenter, task_manager):
        self.segmenter = segmenter
        self.task_manager = task_manager
        self.buffer = ""
        self.processed_segments = 0
        self.current_language = None
        self.trigger_count = 0

    def feed_token(self, token: str, language: str):
        """
        Feed a token and process segments if needed.

        Args:
            token: Text token to add to buffer
            language: Language code ('en' or 'zh'). Must be provided.
        """
        # Validate language
        if language not in self.segmenter.get_available_languages():
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {self.segmenter.get_available_languages()}"
            )

        # Set language if not already set, or validate it matches
        if self.current_language is None:
            self.current_language = language
        elif self.current_language != language:
            # Language change detected - this shouldn't happen in normal usage
            logger.warning(
                f"Language changed from {self.current_language} to {language}. Resetting buffer."
            )
            self.reset()
            self.current_language = language

        self.buffer += token

        # Count actual tokens using spaCy
        token_count = self._count_tokens()

        # Process segments every K tokens
        if token_count // DETECTION_INTERVAL > self.trigger_count:
            self.trigger_count = token_count // DETECTION_INTERVAL
            self._process_segments()

    def _count_tokens(self) -> int:
        """Count actual tokens in buffer using spaCy."""
        if not self.buffer.strip() or self.current_language is None:
            return 0

        # Use appropriate spaCy model to count tokens
        nlp = self.segmenter.models.get(self.current_language)
        if nlp:
            doc = nlp(self.buffer)
            return len([token for token in doc if not token.is_space])
        else:
            # Fallback to simple word count
            return len(self.buffer.split())

    def flush(self) -> int:
        """Process all remaining segments and return count processed."""
        if not self.buffer.strip() or self.current_language is None:
            return 0

        # Clean up the buffer text - normalize spaces like the direct segmenter would see
        cleaned_buffer = " ".join(self.buffer.split())
        current_segments = self.segmenter.segment_sentences(
            cleaned_buffer, language=self.current_language
        )

        if len(current_segments) > self.processed_segments:
            remaining = current_segments[self.processed_segments :]
            voice = self.task_manager.synthesis_core.get_default_voice(
                self.current_language
            )
            self.task_manager.queue_tasks(remaining, self.current_language, voice)

            count = len(remaining)
            self.processed_segments = len(current_segments)
            return count

        return 0

    def _process_segments(self):
        """Process buffer for new complete segments."""
        if not self.buffer.strip() or self.current_language is None:
            return

        # Clean up the buffer text - normalize spaces like the direct segmenter would see
        cleaned_buffer = " ".join(self.buffer.split())
        current_segments = self.segmenter.segment_sentences(
            cleaned_buffer, language=self.current_language
        )

        if len(current_segments) > self.processed_segments:
            new_segments = current_segments[self.processed_segments :]

            # Always keep last segment for next processing unless it's the final flush
            segments_to_process = new_segments[:-1] if len(new_segments) > 1 else []

            if segments_to_process:
                voice = self.task_manager.synthesis_core.get_default_voice(
                    self.current_language
                )
                self.task_manager.queue_tasks(
                    segments_to_process, self.current_language, voice
                )
                self.processed_segments += len(segments_to_process)

    def reset(self):
        """Reset buffer state."""
        self.buffer = ""
        self.processed_segments = 0
        self.current_language = None
        self.trigger_count = 0

    def get_processed_count(self) -> int:
        """Get number of processed segments."""
        return self.processed_segments
