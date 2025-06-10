import pytest

from src.easy_tts_server.segmenter import TextSegmenter, segment_text


class TestTextSegmenter:
    """Test text segmentation functionality."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_english_segmentation(self, segmenter):
        """Test English sentence segmentation."""
        text = "Hello world! This is a test. How are you?"
        sentences = segmenter.segment_text(text, language="en", min_tokens=0)

        assert len(sentences) == 3
        assert sentences[0] == "Hello world!"
        assert sentences[1] == "This is a test."
        assert sentences[2] == "How are you?"

    def test_chinese_segmentation(self, segmenter):
        """Test Chinese sentence segmentation."""
        text = "你好世界！这是测试。你好吗？"
        sentences = segmenter.segment_text(text, language="zh", min_tokens=0)

        assert len(sentences) == 3
        assert sentences[0] == "你好世界！"
        assert sentences[1] == "这是测试。"
        assert sentences[2] == "你好吗？"

    def test_complex_english_segmentation(self, segmenter):
        """Test complex English with abbreviations."""
        text = "Dr. Smith went to the U.S.A. He bought a car for $20,000."
        sentences = segmenter.segment_text(text, language="en")

        assert len(sentences) == 2
        assert "Dr. Smith went to the U.S.A." in sentences[0]
        assert "$20,000" in sentences[1]

    def test_explicit_language_specification(self, segmenter):
        """Test explicit language specification."""
        text = "Hello world! This is a test."

        # Test with explicit English
        en_sentences = segmenter.segment_text(text, language="en", min_tokens=0)
        assert len(en_sentences) == 2

        # Test with explicit Chinese (should still work)
        zh_sentences = segmenter.segment_text(text, language="zh", min_tokens=0)
        assert len(zh_sentences) >= 1  # May segment differently but should work

    def test_single_sentence(self, segmenter):
        """Test single sentence input."""
        assert segmenter.segment_text("Hello world!", language="en", min_tokens=0) == [
            "Hello world!"
        ]
        assert segmenter.segment_text("你好世界！", language="zh", min_tokens=0) == [
            "你好世界！"
        ]

    def test_empty_input(self, segmenter):
        """Test empty input handling."""
        assert segmenter.segment_text("", language="en", min_tokens=0) == []
        assert segmenter.segment_text("   ", language="en", min_tokens=0) == []

    def test_invalid_language(self, segmenter):
        """Test invalid language handling."""
        text = "Hello world!"
        with pytest.raises(ValueError, match="Unsupported language"):
            segmenter.segment_text(text, language="invalid")

    def test_available_languages(self, segmenter):
        """Test available languages."""
        languages = segmenter.get_available_languages()
        assert "en" in languages
        assert "zh" in languages

    def test_model_availability(self, segmenter):
        """Test model availability checks."""
        assert segmenter.is_model_available("en") is True
        assert segmenter.is_model_available("zh") is True
        assert segmenter.is_model_available("fr") is False


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_segment_text_function_english(self):
        """Test the standalone segment_text function with English."""
        text = "This is a test. It works well."
        sentences = segment_text(text, language="en", min_tokens=0)

        assert len(sentences) == 2
        assert sentences[0] == "This is a test."
        assert sentences[1] == "It works well."

    def test_segment_text_function_chinese(self):
        """Test the standalone segment_text function with Chinese."""
        text = "你好世界！这是测试。"
        sentences = segment_text(text, language="zh", min_tokens=0)

        assert len(sentences) == 2
        assert sentences[0] == "你好世界！"
        assert sentences[1] == "这是测试。"


class TestIntegration:
    """Integration tests for segmentation functionality."""

    def test_explicit_language_english(self):
        """Test explicit language specification with English text."""
        text = "Hello world! This is a test sentence."
        sentences = segment_text(text, language="en", min_tokens=0)

        assert len(sentences) == 2
        assert all(isinstance(s, str) for s in sentences)

    def test_explicit_language_chinese(self):
        """Test explicit language specification with Chinese text."""
        text = "你好世界！这是测试句子。"
        sentences = segment_text(text, language="zh", min_tokens=0)

        assert len(sentences) == 2
        assert all(isinstance(s, str) for s in sentences)

    def test_robustness(self):
        """Test robustness with various edge cases."""
        edge_cases_en = [
            "",
            "   ",
            "Single.",
            "No punctuation here",
            "Multiple!!! Exclamations???",
        ]

        edge_cases_zh = [
            "单个。",
            "没有标点符号",
            "多个！！！感叹号？？？",
            "Mixed中文English句子。",
        ]

        # Test English cases
        for text in edge_cases_en:
            sentences = segment_text(text, language="en", min_tokens=0)
            assert isinstance(sentences, list)
            assert all(isinstance(s, str) for s in sentences)

            if text.strip():  # Non-empty text should produce at least one sentence
                assert len(sentences) >= 1

        # Test Chinese cases
        for text in edge_cases_zh:
            sentences = segment_text(text, language="zh", min_tokens=0)
            assert isinstance(sentences, list)
            assert all(isinstance(s, str) for s in sentences)

            if text.strip():  # Non-empty text should produce at least one sentence
                assert len(sentences) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
