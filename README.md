# Easy TTS Server

[![Test Suite](https://github.com/3rd-Musketeer/easy-tts-server/workflows/Test%20Suite/badge.svg)](https://github.com/3rd-Musketeer/easy-tts-server/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple and efficient text-to-speech server with streaming capabilities, supporting both English and Chinese languages.

## ✨ Features

- **Multi-language Support**: English and Chinese TTS with high-quality voices
- **Streaming Synthesis**: Real-time audio generation for live applications
- **Smart Text Processing**: Intelligent text segmentation and normalization
- **Background Processing**: Non-blocking synthesis with threading
- **Markdown Support**: Automatic markdown cleanup for clean TTS output
- **Token-based Streaming**: Feed tokens incrementally for LLM integration
- **Voice Selection**: Multiple voice options for each language

## 🚀 Quick Start

### Installation

Install directly from GitHub (recommended):

```bash
# Using pip
pip install git+https://github.com/3rd-Musketeer/easy-tts-server.git

# Or using uv (recommended for faster installs)
uv add git+https://github.com/3rd-Musketeer/easy-tts-server.git
```

Alternatively, for development or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/3rd-Musketeer/easy-tts-server.git
cd easy-tts-server

# Install using uv (recommended)
uv sync
uv pip install -e .

# Or using pip
pip install -e .
```

### Dependencies

The package requires spaCy language models:

```bash
# Download required spaCy models
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

### Basic Usage

```python
from easy_tts_server import create_tts_engine

# Create TTS engine
tts = create_tts_engine()

# Simple text-to-speech
audio = tts.tts("Hello, world!", language="en")

# With voice selection
audio = tts.tts("Hello, world!", language="en", voice="af_maple")

# Chinese TTS
audio = tts.tts("你好，世界！", language="zh", voice="zf_001")
```

## 📖 API Documentation

### TTSEngine

The main TTS engine class providing synthesis capabilities.

#### Methods

##### `tts(text: str, language: str, voice: str = None) -> np.ndarray`

Synthesize entire text to audio.

**Parameters:**
- `text`: Input text to synthesize
- `language`: Language code (`'en'` or `'zh'`)
- `voice`: Voice name (optional, uses default if not specified)

**Returns:** Audio as numpy array

##### `tts_stream(text: str, language: str, voice: str = None) -> Generator[np.ndarray, None, None]`

Stream audio synthesis with background processing.

**Parameters:**
- `text`: Input text to synthesize
- `language`: Language code (`'en'` or `'zh'`)
- `voice`: Voice name (optional)

**Yields:** Audio chunks as numpy arrays

##### `feed_in(token: str, language: str)`

Feed tokens for streaming synthesis (useful for LLM integration).

**Parameters:**
- `token`: Text token to process
- `language`: Language code (`'en'` or `'zh'`)

##### `feed_out() -> Generator[np.ndarray, None, None]`

Stream audio segments as they become available.

**Yields:** Audio chunks as numpy arrays

##### `flush()`

Process any remaining content in buffer and complete synthesis.

##### `reset()`

Reset all TTS operations and cancel ongoing synthesis.

#### Properties

- `sample_rate`: Audio sample rate (24000 Hz)
- Available languages: `['en', 'zh']`

### Available Voices

#### English Voices
- `af_maple` (default) - American Female
- `af_sol` - American Female  
- `bf_vale` - British Female

#### Chinese Voices
- `zf_001` (default) - Chinese Female
- `zm_010` - Chinese Male

## 💡 Usage Examples

### Basic Text-to-Speech

```python
from easy_tts_server import create_tts_engine
import soundfile as sf

# Initialize engine
tts = create_tts_engine()

# Generate English speech
audio = tts.tts("Welcome to Easy TTS Server!", language="en")
sf.write("output.wav", audio, tts.sample_rate)

# Generate Chinese speech
audio = tts.tts("欢迎使用简易语音合成服务器！", language="zh")
sf.write("output_zh.wav", audio, tts.sample_rate)
```

### Streaming Synthesis

```python
import sounddevice as sd

# Stream audio in real-time
def play_stream(text, language):
    for audio_chunk in tts.tts_stream(text, language):
        sd.play(audio_chunk, tts.sample_rate)
        sd.wait()

play_stream("This is streaming text-to-speech synthesis.", "en")
```

### LLM Integration with Token Feeding

```python
# Simulate streaming from an LLM
def simulate_llm_stream():
    tokens = ["Hello", " there", "!", " How", " are", " you", " today", "?"]
    
    for token in tokens:
        tts.feed_in(token, language="en")
        
        # Get available audio
        for audio_chunk in tts.feed_out():
            sd.play(audio_chunk, tts.sample_rate)
    
    # Complete the synthesis
    tts.flush()
    for audio_chunk in tts.feed_out():
        sd.play(audio_chunk, tts.sample_rate)

simulate_llm_stream()
```

### Text Segmentation

```python
from easy_tts_server import segment_text

# Intelligent text segmentation
text = "Hello world. This is a test. How are you?"
segments = segment_text(text, language="en", min_tokens=5)
print(segments)
# Output: ['Hello world. This is a test.', 'How are you?']
```

### Voice Selection

```python
# List available voices
print("English voices:", tts.get_voices("en"))
print("Chinese voices:", tts.get_voices("zh"))

# Use different voices
audio1 = tts.tts("Hello!", language="en", voice="af_maple")
audio2 = tts.tts("Hello!", language="en", voice="bf_vale")
```

## 🏗️ Architecture

### Core Components

- **TTSEngine**: Main API interface
- **SynthesisCore**: Low-level model operations using Kokoro TTS
- **TaskManager**: Background threading and task queue management
- **StreamingBuffer**: Token-based streaming and dynamic segmentation
- **TextSegmenter**: Intelligent text segmentation using spaCy
- **Utils**: Text normalization and preprocessing

### Threading Model

The engine uses background threading for non-blocking synthesis:
- Main thread handles API calls
- Worker thread processes synthesis tasks
- Audio is streamed as it becomes available

## 🔧 Configuration

### Model Selection

By default, the engine uses `hexgrad/Kokoro-82M-v1.1-zh`. You can specify a different model:

```python
from easy_tts_server import TTSEngine

# Use a different model
tts = TTSEngine(repo_id="your-custom-model")
```

### Text Processing

The engine automatically:
- Normalizes Unicode text
- Removes markdown formatting
- Segments text intelligently
- Handles mixed Chinese-English text

## 📋 Requirements

- Python ≥ 3.10
- PyTorch (CPU or CUDA)
- spaCy with language models
- Kokoro TTS
- NumPy
- SoundFile (for audio I/O)
- SoundDevice (for audio playback)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is open source. Please check the license file for details.

## 🙏 Acknowledgments

- Built on [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh) for high-quality synthesis
- Uses [spaCy](https://spacy.io/) for intelligent text processing
- Inspired by the need for simple, efficient TTS solutions

---

**Easy TTS Server** - Making text-to-speech simple and efficient! 🎵
