from typing import Optional, Tuple
import re
import unicodedata

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
    chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
    return bool(chinese_pattern.search(text))

def normalize_text(text: str) -> str:
    """
    Normalize text by removing markdown decorators while keeping the content.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text with markdown decorators removed
    """
    if not text:
        return text
    
    # Unicode normalization - convert to NFC (Canonical Decomposition, followed by Canonical Composition)
    # This handles composed/decomposed characters and ensures consistent representation
    normalized = unicodedata.normalize('NFC', text)
    
    # Remove markdown decorators but keep the content
    # Remove headers (# ## ###) 
    normalized = re.sub(r'^#{1,6}\s+', '', normalized, flags=re.MULTILINE)
    
    # Remove emphasis decorators (**bold**, *italic*, __bold__, _italic_)
    normalized = re.sub(r'\*\*([^*]+)\*\*', r'\1', normalized)
    normalized = re.sub(r'\*([^*]+)\*', r'\1', normalized)
    normalized = re.sub(r'__([^_]+)__', r'\1', normalized)
    normalized = re.sub(r'_([^_]+)_', r'\1', normalized)
    
    # Remove strikethrough ~~text~~
    normalized = re.sub(r'~~([^~]+)~~', r'\1', normalized)
    
    # Remove code blocks and inline code
    normalized = re.sub(r'```[\s\S]*?```', '', normalized)
    normalized = re.sub(r'`([^`]*)`', r'\1', normalized)
    
    # Remove links but keep text [text](url) -> text
    normalized = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', normalized)
    
    # Remove images ![alt](url)
    normalized = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', normalized)
    
    # Remove list markers (- * + and numbered lists)
    normalized = re.sub(r'^[\s]*[-*+]\s+', '', normalized, flags=re.MULTILINE)
    normalized = re.sub(r'^\s*\d+\.\s+', '', normalized, flags=re.MULTILINE)
    
    # Remove blockquote markers (>)
    normalized = re.sub(r'^\s*>\s*', '', normalized, flags=re.MULTILINE)
    
    # Basic cleanup
    # Preserve paragraph breaks (double newlines) but clean up excessive spacing
    normalized = re.sub(r'\n\s*\n\s*\n+', '\n\n', normalized)  # Max 2 consecutive newlines
    normalized = re.sub(r'[ \t]+', ' ', normalized)  # Clean horizontal whitespace but keep \n
    normalized = re.sub(r' *\n *', '\n', normalized)  # Clean spaces around newlines
    normalized = normalized.strip()
    
    return normalized

def preprocess_text(text: str, language: str) -> Tuple[str, str]:
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