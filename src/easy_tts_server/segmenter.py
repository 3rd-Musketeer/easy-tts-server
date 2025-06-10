import logging

import spacy

logger = logging.getLogger(__name__)


class TextSegmenter:
    """
    A text segmenter that segments text into sentences
    using appropriate spaCy models for Chinese and English.
    Language must be explicitly provided by the user.
    """

    def __init__(self):
        self.models = {}
        self.supported_languages = ["en", "zh"]
        self.min_tokens = 3
        self._load_models()

    def _load_models(self):
        """Load spaCy models for supported languages."""
        model_names = {"en": "en_core_web_sm", "zh": "zh_core_web_sm"}

        for lang, model_name in model_names.items():
            try:
                self.models[lang] = spacy.load(model_name)
                logger.info(f"Loaded {model_name} for {lang}")
            except OSError:
                logger.warning(
                    f"Model {model_name} not found. Please install it with: python -m spacy download {model_name}"
                )
                # Fallback to blank model with sentencizer
                try:
                    nlp = spacy.blank(lang)
                    nlp.add_pipe("sentencizer")
                    self.models[lang] = nlp
                    logger.info(f"Using blank model with sentencizer for {lang}")
                except Exception as e:
                    logger.error(f"Failed to create fallback model for {lang}: {e}")

    def segment_sentences(
        self, text: str, language: str, min_tokens: int | None = None
    ) -> list[str]:
        """
        Segment text into sentences using the appropriate spaCy model, ensuring each segment
        has at least min_tokens tokens for proper TTS model performance.

        Args:
            text: Input text to segment
            language: Language code ('en' or 'zh'). Must be provided.
            min_tokens: Minimum number of tokens per segment. If None, uses self.min_tokens (default: 5)

        Returns:
            List of sentence strings, each with at least min_tokens tokens
        """
        if not text.strip():
            return []

        # Use class default if min_tokens not provided
        if min_tokens is None:
            min_tokens = self.min_tokens

        # Validate language
        if language not in self.supported_languages:
            raise ValueError(
                f"Unsupported language: {language}. Supported languages: {self.supported_languages}"
            )

        # Check if model is available
        if language not in self.models:
            logger.error(f"No model available for language: {language}")
            return [text]  # Return original text as single sentence

        try:
            # Process text with spaCy
            nlp = self.models[language]
            doc = nlp(text)

            # Extract individual sentences first
            raw_sentences = []
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                if sentence_text:  # Only add non-empty sentences
                    raw_sentences.append(sentence_text)

            # If no sentences were detected, return original text
            if not raw_sentences:
                return [text.strip()]

            # Combine sentences to meet minimum token requirement
            combined_segments = []
            current_segment = ""
            current_token_count = 0

            for sentence in raw_sentences:
                # Count tokens in this sentence
                sentence_doc = nlp(sentence)
                sentence_tokens = len(
                    [token for token in sentence_doc if not token.is_space]
                )

                # If current segment is empty, start with this sentence
                if not current_segment:
                    current_segment = sentence
                    current_token_count = sentence_tokens
                else:
                    # Add sentence to current segment
                    current_segment += " " + sentence
                    current_token_count += sentence_tokens

                # If we have enough tokens, finalize this segment
                if current_token_count >= min_tokens:
                    combined_segments.append(current_segment)
                    current_segment = ""
                    current_token_count = 0

            # Handle any remaining segment
            if current_segment:
                if combined_segments:
                    # If it's too short, merge with the last segment
                    if current_token_count < min_tokens:
                        combined_segments[-1] += " " + current_segment
                    else:
                        combined_segments.append(current_segment)
                else:
                    # If it's the only segment, keep it regardless of length
                    combined_segments.append(current_segment)

            return combined_segments

        except Exception as e:
            logger.error(f"Error during sentence segmentation: {e}")
            return [text]  # Return original text as fallback

    def segment_text(
        self, text: str, language: str, min_tokens: int | None = None
    ) -> list[str]:
        """
        Main method to segment text into sentences.
        Alias for segment_sentences for backward compatibility.
        """
        return self.segment_sentences(text, language, min_tokens)

    def get_available_languages(self) -> list[str]:
        """Return list of available languages based on loaded models."""
        return list(self.models.keys())

    def is_model_available(self, language: str) -> bool:
        """Check if a model is available for the given language."""
        return language in self.models


# Convenience function for easy usage
def segment_text(text: str, language: str, min_tokens: int | None = None) -> list[str]:
    """
    Convenience function to segment text into sentences.

    Args:
        text: Input text to segment
        language: Language code ('en' or 'zh'). Must be provided.
        min_tokens: Minimum number of tokens per segment. If None, uses default (5)

    Returns:
        List of sentence strings, each with at least min_tokens tokens
    """
    segmenter = TextSegmenter()
    return segmenter.segment_sentences(text, language, min_tokens)


if __name__ == "__main__":
    # Example usage
    segmenter = TextSegmenter()

    # Test English text
    en_text = "Hello world. This is a test sentence. How are you doing today?"
    en_sentences = segmenter.segment_text(en_text, "en")
    print("English sentences:")
    for i, sent in enumerate(en_sentences, 1):
        print(f"{i}: {sent}")

    # Test Chinese text
    zh_text = "你好世界。这是一个测试句子。你今天过得怎么样？"
    zh_sentences = segmenter.segment_text(zh_text, "zh")
    print("\nChinese sentences:")
    for i, sent in enumerate(zh_sentences, 1):
        print(f"{i}: {sent}")
