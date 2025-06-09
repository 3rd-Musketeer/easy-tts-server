"""
Easy TTS Server - A simple and efficient text-to-speech server.

This package provides:
- TTSEngine: Main TTS engine with streaming capabilities
- TextSegmenter: Language detection and text segmentation
- Utility functions for language detection and audio processing
"""

from .tts import TTSEngine, create_tts_engine
from .segmenter import TextSegmenter, segment_text

__all__ = [
    'TTSEngine',
    'create_tts_engine', 
    'TextSegmenter',
    'segment_text'
]

__version__ = '0.1.0' 