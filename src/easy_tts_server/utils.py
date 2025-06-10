import re
import unicodedata

import markdown
from bs4 import BeautifulSoup


def has_chinese_characters(text: str) -> bool:
    """
    Check if text contains any Chinese characters.

    Args:
        text: Input text to check

    Returns:
        True if Chinese characters are found, False otherwise
    """
    # Chinese character ranges:
    # \u4e00-\u9fff: CJK Unified Ideographs (main Chinese characters)
    # \u3400-\u4dbf: CJK Extension A
    # \uf900-\ufaff: CJK Compatibility Ideographs
    chinese_pattern = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
    return bool(chinese_pattern.search(text))


def normalize_text(text: str) -> str:
    """
    Normalize text by parsing markdown and extracting clean text content.

    Args:
        text: Input text to normalize (can contain markdown)

    Returns:
        Normalized text with markdown converted to plain text
    """
    if not text:
        return text

    # Unicode normalization - convert to NFC (Canonical Decomposition, followed by Canonical Composition)
    # This handles composed/decomposed characters and ensures consistent representation
    normalized = unicodedata.normalize("NFC", text)

    try:
        # Parse markdown to HTML using the markdown library
        # This properly handles all markdown syntax including headers, lists, links, etc.
        html = markdown.markdown(
            normalized,
            extensions=[
                "tables",  # Support for tables
                "fenced_code",  # Support for fenced code blocks
                "toc",  # Table of contents
                "sane_lists",  # Better list handling
            ],
        )

        # Use BeautifulSoup to extract clean text from HTML
        soup = BeautifulSoup(html, "html.parser")

        # Remove code blocks and inline code completely
        for code_element in soup.find_all(["code", "pre"]):
            code_element.decompose()

        # Extract text content preserving structure
        # Use separator='\n' for block elements to preserve line breaks
        clean_text = soup.get_text(separator="\n", strip=True)

    except Exception:
        # Fallback to original text if markdown parsing fails
        clean_text = normalized

    # Basic cleanup while preserving structure
    # Clean up excessive whitespace but preserve intentional line breaks
    clean_text = re.sub(
        r"[ \t]+", " ", clean_text
    )  # Replace multiple spaces/tabs with single space
    clean_text = re.sub(r" *\n *", "\n", clean_text)  # Clean spaces around newlines
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)  # Max 2 consecutive newlines
    clean_text = clean_text.strip()

    return clean_text


def preprocess_text(text: str, language: str) -> tuple[str, str]:
    """
    Preprocess text with normalization.

    Args:
        text: Input text to preprocess
        language: Language code ('en' or 'zh')

    Returns:
        Tuple of (normalized_text, language)
    """
    if not text:
        return "", language

    # Normalize the text
    normalized_text = normalize_text(text)

    return normalized_text, language
