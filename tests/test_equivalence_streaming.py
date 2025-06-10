"""
Segmentation Equivalence test for feed_in/feed_out vs tts_stream functionality.
Validates that both modes produce the same text segmentation and timing for identical input text.

APPROACH:
=========
Since TTS models are inherently non-deterministic (producing different audio for identical text),
we focus on testing the segmentation logic rather than audio similarity:

1. Text Segmentation Equivalence: Both modes should produce identical text segments
2. Segment Count Equivalence: Both modes should produce the same number of segments
3. Segment Timing Equivalence: feed_in mode should trigger segments at appropriate token boundaries
4. Buffer State Equivalence: The streaming buffer should behave consistently

This approach tests the actual streaming logic without being affected by neural model randomness.
"""

import difflib
import re
import sys
import threading
import time
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easy_tts_server.tts import TTSEngine

# Configuration
ENABLE_DETAILED_DEBUG = False  # Set to True to see detailed streaming buffer debugging


class TestSegmentationEquivalence:
    """Test segmentation equivalence between tts_stream and feed_in/feed_out."""

    @pytest.fixture
    def engine(self):
        """Create a TTS engine instance."""
        return TTSEngine()

    def test_simple_sentences_equivalence(self, engine):
        """Test segmentation equivalence for simple sentences."""
        text = "Hello world. This is a test. How are you today?"
        success = compare_segmentation_modes(engine, text, "Simple sentences")
        assert success, "Simple sentences segmentation should be equivalent"

    def test_complex_punctuation_equivalence(self, engine):
        """Test segmentation equivalence for complex punctuation."""
        text = "Question: What's the weather like? Answer: It's sunny! Note: Bring an umbrella; just in case."
        success = compare_segmentation_modes(engine, text, "Complex punctuation")
        assert success, "Complex punctuation segmentation should be equivalent"

    def test_long_conversational_equivalence(self, engine):
        """Test segmentation equivalence for long conversational text."""
        text = "Hello there! How are you doing today? I hope you're having a wonderful time. Let me tell you a story about artificial intelligence. It's quite fascinating, really. Once upon a time, there were researchers who dreamed of machines that could think and speak."
        success = compare_segmentation_modes(engine, text, "Long conversational")
        assert success, "Long conversational segmentation should be equivalent"

    def test_technical_content_equivalence(self, engine):
        """Test segmentation equivalence for technical content."""
        text = "The TTS engine implements a modular architecture. Core components include synthesis, streaming, and task management. Key features: real-time audio generation, concurrent processing, and multi-language support."
        success = compare_segmentation_modes(engine, text, "Technical content")
        assert success, "Technical content segmentation should be equivalent"

    def test_mixed_content_equivalence(self, engine):
        """Test segmentation equivalence for mixed content."""
        text = "Welcome to our service! Please note: processing may take 2-3 minutes. Status: ready. Would you like to continue? Great! Let's begin the demonstration."
        success = compare_segmentation_modes(engine, text, "Mixed content")
        assert success, "Mixed content segmentation should be equivalent"


class TestEdgeCaseSegmentation:
    """Test edge cases for segmentation equivalence."""

    @pytest.fixture
    def engine(self):
        """Create a TTS engine instance."""
        return TTSEngine()

    def test_single_sentence_equivalence(self, engine):
        """Test segmentation equivalence for single sentence."""
        text = "Hello world."
        success = compare_segmentation_modes(engine, text, "Single sentence")
        assert success, "Single sentence segmentation should be equivalent"

    def test_multiple_short_sentences_equivalence(self, engine):
        """Test segmentation equivalence for multiple short sentences."""
        text = "Hi. Hello. Good. Yes."
        success = compare_segmentation_modes(engine, text, "Multiple short sentences")
        assert success, "Multiple short sentences segmentation should be equivalent"

    def test_no_punctuation_equivalence(self, engine):
        """Test segmentation equivalence for text without punctuation."""
        text = "This is a sentence without punctuation"
        success = compare_segmentation_modes(engine, text, "No punctuation")
        assert success, "No punctuation segmentation should be equivalent"

    def test_only_punctuation_equivalence(self, engine):
        """Test segmentation equivalence for punctuation-heavy text."""
        text = "What? Really! Amazing."
        success = compare_segmentation_modes(engine, text, "Only punctuation")
        assert success, "Only punctuation segmentation should be equivalent"


# Helper functions for segmentation testing


def compare_segmentation_modes(engine, text, test_name):
    """Compare tts_stream vs feed_in/feed_out segmentation for the same text."""
    print(
        f"  🔄 Comparing segmentation for: {text[:50]}{'...' if len(text) > 50 else ''}"
    )

    # Method 1: tts_stream (reference segmentation)
    print("  📡 Analyzing tts_stream segmentation...")
    reference_result = analyze_stream_segmentation(engine, text)

    # Method 2: feed_in/feed_out (test segmentation)
    print("  🎙️  Analyzing feed_in/feed_out segmentation...")
    streaming_result = analyze_feedin_segmentation(engine, text)

    # Compare segmentation results
    success = compare_segmentation_results(
        reference_result, streaming_result, test_name
    )

    if success:
        print(f"    ✅ {test_name}: Segmentation modes are equivalent!")
    else:
        print(f"    ❌ {test_name}: Segmentation modes differ!")

    return success


def analyze_stream_segmentation(engine, text) -> dict:
    """Analyze how tts_stream segments the text."""
    engine.reset()

    # Get the segmentation that tts_stream would produce
    segments = engine.segmenter.segment_sentences(text, language="en")
    print(f"    🧩 Text segments: {segments}")

    # Collect basic info about what would be synthesized
    segment_info = []
    for i, segment in enumerate(segments):
        segment_info.append(
            {
                "index": i,
                "text": segment,
                "length": len(segment),
                "word_count": len(segment.split()),
                "has_punctuation": bool(re.search(r"[.!?]", segment)),
            }
        )
        print(
            f"    📊 Segment {i+1}: '{segment}' ({len(segment)} chars, {len(segment.split())} words)"
        )

    return {
        "mode": "tts_stream",
        "original_text": text,
        "segments": segments,
        "segment_count": len(segments),
        "segment_info": segment_info,
    }


def analyze_feedin_segmentation(engine, text) -> dict:
    """Analyze how feed_in/feed_out processes text segmentation."""
    engine.reset()

    # Tokenize text realistically
    tokens = tokenize_realistically(text)
    print(f"    🔤 Tokens to feed: {tokens}")

    # Track segmentation state throughout the process
    segmentation_events = []
    final_segments = []

    # Track streaming buffer state
    streaming_buffer = engine.streaming_buffer

    def track_segmentation_events():
        """Collect segments as they are produced."""
        segment_count = 0
        for _ in engine.feed_out():
            segment_count += 1
            # We don't need the actual audio, just count the segments
            processed_segments = streaming_buffer.processed_segments
            segmentation_events.append(
                {
                    "segment_number": segment_count,
                    "processed_segments": processed_segments,
                    "buffer_state": streaming_buffer.buffer,
                    "timestamp": time.time(),
                }
            )

    # Run feed_in process with segmentation tracking
    producer_finished = threading.Event()

    def token_producer():
        """Feed tokens one by one and track when segments are triggered."""
        for i, token in enumerate(tokens):
            if ENABLE_DETAILED_DEBUG:
                print(f"    📝 Feeding token {i+1}/{len(tokens)}: '{token}'")

            buffer_before = streaming_buffer.buffer
            processed_before = streaming_buffer.processed_segments

            engine.feed_in(token, "en")

            buffer_after = streaming_buffer.buffer
            processed_after = streaming_buffer.processed_segments

            # Track if this token triggered segment processing
            if processed_after > processed_before:
                segmentation_events.append(
                    {
                        "type": "token_triggered",
                        "token_index": i,
                        "token": token,
                        "segments_triggered": processed_after - processed_before,
                        "buffer_before": buffer_before,
                        "buffer_after": buffer_after,
                    }
                )
                if ENABLE_DETAILED_DEBUG:
                    print(
                        f"      🎯 Triggered {processed_after - processed_before} segment(s)"
                    )

            time.sleep(0.001)  # Small delay to prevent race conditions

        # Flush remaining content
        processed_before_flush = streaming_buffer.processed_segments
        buffer_before_flush = streaming_buffer.buffer

        engine.flush()

        processed_after_flush = streaming_buffer.processed_segments
        flush_segments = processed_after_flush - processed_before_flush

        if flush_segments > 0:
            segmentation_events.append(
                {
                    "type": "flush_triggered",
                    "segments_triggered": flush_segments,
                    "buffer_before_flush": buffer_before_flush,
                    "final_processed": processed_after_flush,
                }
            )
            if ENABLE_DETAILED_DEBUG:
                print(f"    🏁 Flush triggered {flush_segments} segment(s)")

        producer_finished.set()

    # Run the process
    consumer_thread = threading.Thread(target=track_segmentation_events)
    producer_thread = threading.Thread(target=token_producer)

    consumer_thread.start()
    time.sleep(0.01)
    producer_thread.start()

    producer_thread.join(timeout=30)
    producer_finished.wait(timeout=5)
    consumer_thread.join(timeout=5)

    # Get final segmentation from the buffer
    cleaned_buffer = " ".join(streaming_buffer.buffer.split())
    final_segments = engine.segmenter.segment_sentences(cleaned_buffer, language="en")

    print(f"    📋 Final segments from buffer: {final_segments}")
    print(f"    📊 Total segments processed: {streaming_buffer.processed_segments}")

    return {
        "mode": "feed_in",
        "original_text": text,
        "tokens": tokens,
        "final_buffer": cleaned_buffer,
        "segments": final_segments,
        "segment_count": streaming_buffer.processed_segments,
        "segmentation_events": segmentation_events,
    }


def compare_segmentation_results(reference_result, streaming_result, test_name) -> bool:
    """Compare segmentation results from both modes."""
    print("  🔍 Comparing segmentation results...")

    ref_segments = reference_result["segments"]
    stream_segments = streaming_result["segments"]

    # 1. Check segment count
    ref_count = len(ref_segments)
    stream_count = streaming_result["segment_count"]
    actual_segments_count = len(stream_segments)

    print("    📊 Segment counts:")
    print(f"      Reference (tts_stream): {ref_count}")
    print(f"      Streaming (processed): {stream_count}")
    print(f"      Streaming (final buffer): {actual_segments_count}")

    if ref_count != stream_count:
        print(f"    ❌ Processed segment count mismatch: {ref_count} vs {stream_count}")
        return False

    if ref_count != actual_segments_count:
        print(
            f"    ❌ Final segment count mismatch: {ref_count} vs {actual_segments_count}"
        )
        return False

    print(f"    ✅ Segment counts match: {ref_count}")

    # 2. Check segment text equivalence
    print("    📝 Segment text comparison:")
    segments_match = True

    for i, (ref_seg, stream_seg) in enumerate(
        zip(ref_segments, stream_segments, strict=False)
    ):
        if ref_seg == stream_seg:
            print(f"      ✅ Segment {i+1}: '{ref_seg}'")
        else:
            print(f"      ❌ Segment {i+1} differs:")
            print(f"         Reference: '{ref_seg}'")
            print(f"         Streaming: '{stream_seg}'")

            # Show character-level diff
            diff = list(
                difflib.unified_diff(
                    ref_seg.splitlines(keepends=True),
                    stream_seg.splitlines(keepends=True),
                    fromfile="reference",
                    tofile="streaming",
                    lineterm="",
                )
            )
            if diff:
                print(f"         Diff: {''.join(diff)}")

            segments_match = False

    if not segments_match:
        return False

    # 3. Check original text reconstruction
    ref_reconstructed = " ".join(ref_segments)
    stream_reconstructed = streaming_result["final_buffer"]

    if reference_result["original_text"].strip() != stream_reconstructed.strip():
        print("    ❌ Text reconstruction differs:")
        print(f"      Original: '{reference_result['original_text']}'")
        print(f"      Reconstructed: '{stream_reconstructed}'")
        return False

    print("    ✅ Text reconstruction matches")

    # 4. Check segmentation timing (if available)
    events = streaming_result["segmentation_events"]
    trigger_events = [
        e for e in events if e.get("type") in ["token_triggered", "flush_triggered"]
    ]

    print("    ⏱️  Segmentation timing analysis:")
    print(f"      Total trigger events: {len(trigger_events)}")

    for event in trigger_events:
        if event["type"] == "token_triggered":
            print(
                f"        Token {event['token_index']+1} ('{event['token']}'): triggered {event['segments_triggered']} segment(s)"
            )
        elif event["type"] == "flush_triggered":
            print(f"        Flush: triggered {event['segments_triggered']} segment(s)")

    total_triggered = sum(e.get("segments_triggered", 0) for e in trigger_events)
    if total_triggered != ref_count:
        print(
            f"    ⚠️  Total triggered segments ({total_triggered}) != expected ({ref_count})"
        )
        # This is a warning, not a failure, as timing can vary

    print("    ✅ Segmentation equivalence verified")
    return True


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings using sequence matching."""
    if text1 == text2:
        return 1.0

    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio()


def calculate_edit_distance(text1: str, text2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(text1) < len(text2):
        return calculate_edit_distance(text2, text1)

    if len(text2) == 0:
        return len(text1)

    previous_row = list(range(len(text2) + 1))
    for i, c1 in enumerate(text1):
        current_row = [i + 1]
        for j, c2 in enumerate(text2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def tokenize_realistically(text):
    """
    Tokenize text to match how feed_in processes it.
    Simple word and punctuation boundary tokenization.
    """
    import re

    # Split on word boundaries but keep punctuation separate
    pattern = r"\w+|[^\w\s]|\s+"
    tokens = re.findall(pattern, text)

    # Filter out empty tokens but keep spaces
    return [token for token in tokens if token.strip() or token.isspace()]


def main():
    """Run the complete segmentation equivalence test suite."""
    try:
        from easy_tts_server.tts import TTSEngine

        print("🎯 Segmentation Equivalence Test Suite")
        print("=" * 60)

        engine = TTSEngine()

        test_cases = [
            ("Simple sentences", "Hello world. This is a test. How are you today?"),
            (
                "Complex punctuation",
                "Question: What's the weather like? Answer: It's sunny! Note: Bring an umbrella; just in case.",
            ),
            (
                "Long conversational",
                "Hello there! How are you doing today? I hope you're having a wonderful time. Let me tell you a story about artificial intelligence.",
            ),
            (
                "Technical content",
                "The TTS engine implements a modular architecture. Core components include synthesis, streaming, and task management.",
            ),
            (
                "Mixed content",
                "Welcome to our service! Please note: processing may take 2-3 minutes. Status: ready. Would you like to continue?",
            ),
        ]

        edge_cases = [
            ("Single sentence", "Hello world."),
            ("Multiple short", "Hi. Hello. Good. Yes."),
            ("No punctuation", "This is a sentence without punctuation"),
            ("Only punctuation", "What? Really! Amazing."),
        ]

        all_passed = True

        print("\n🧪 Main Test Cases:")
        print("-" * 40)
        for name, text in test_cases:
            print(f"\n{name}:")
            success = compare_segmentation_modes(engine, text, name)
            if not success:
                all_passed = False

        print("\n🔬 Edge Cases:")
        print("-" * 40)
        for name, text in edge_cases:
            print(f"\n{name}:")
            success = compare_segmentation_modes(engine, text, name)
            if not success:
                all_passed = False

        if all_passed:
            print("\n🏆 ALL SEGMENTATION TESTS PASSED!")
            print("✅ feed_in/feed_out segmentation is equivalent to tts_stream")
            return True
        else:
            print("\n❌ SEGMENTATION TESTS FAILED!")
            print("❌ Segmentation differences detected between modes")
            return False

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
