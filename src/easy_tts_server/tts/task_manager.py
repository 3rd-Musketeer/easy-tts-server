import queue
import threading
from dataclasses import dataclass
from typing import Generator, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class SynthesisTask:
    text: str
    language: str
    voice: str
    task_id: int


class TaskManager:
    """Core task queue + background worker + audio output."""
    
    def __init__(self, synthesis_core):
        self.synthesis_core = synthesis_core
        self.task_queue = queue.Queue()
        self.audio_queue = queue.Queue(maxsize=3)
        self.cancel_flag = threading.Event()
        self.worker_thread = None
        self.task_counter = 0
        self.tasks_completed = 0
        self.streaming_mode = False  # True for feed_in/feed_out, False for tts_stream
        self._lock = threading.Lock()
    
    def queue_task(self, text: str, language: str, voice: str):
        """Queue a single synthesis task."""
        if self.cancel_flag.is_set():
            return
        
        task = SynthesisTask(text, language, voice, self.task_counter)
        self.task_counter += 1
        self.task_queue.put(task)
        self._ensure_worker_running()
    
    def queue_tasks(self, segments: List[str], language: str, voice: str):
        """Queue multiple synthesis tasks."""
        for segment in segments:
            self.queue_task(segment, language, voice)
    
    def set_streaming_mode(self, streaming: bool):
        """Set whether we're in streaming mode (feed_in/feed_out) or batch mode (tts_stream)."""
        self.streaming_mode = streaming
    
    def get_audio(self) -> Generator[np.ndarray, None, None]:
        """Stream audio as it becomes available."""
        while True:
            try:
                if self.cancel_flag.is_set():
                    break
                audio = self.audio_queue.get(timeout=1.0)
                yield audio
            except queue.Empty:
                if self._should_terminate():
                    break
    
    def _should_terminate(self) -> bool:
        """Determine if audio generator should terminate."""
        if self.streaming_mode:
            # In streaming mode, only terminate on cancel or explicit signal
            return self.cancel_flag.is_set()
        else:
            # In batch mode (tts_stream), terminate when all tasks are complete
            return self._all_tasks_complete()
    
    def _all_tasks_complete(self) -> bool:
        """Check if all queued tasks have been processed."""
        with self._lock:
            # All tasks complete if: no pending tasks, no active worker, and queues are empty
            no_pending_tasks = self.task_queue.empty()
            no_active_worker = not (self.worker_thread and self.worker_thread.is_alive())
            no_pending_audio = self.audio_queue.empty()
            
            return no_pending_tasks and no_active_worker and no_pending_audio
    
    def _ensure_worker_running(self):
        """Start worker thread if not already running."""
        with self._lock:
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.worker_thread = threading.Thread(target=self._worker, daemon=True)
                self.worker_thread.start()
    
    def _worker(self):
        """Background worker that processes synthesis tasks."""
        while not self.cancel_flag.is_set():
            try:
                task = self.task_queue.get(timeout=1.0)
                if self.cancel_flag.is_set():
                    break
                
                audio = self.synthesis_core.synthesize(task.text, task.language, task.voice)
                if not self.cancel_flag.is_set():
                    self.audio_queue.put(audio)
                    with self._lock:
                        self.tasks_completed += 1
                    logger.debug(f"Completed task {task.task_id}: {task.text[:30]}...")
                    
            except queue.Empty:
                # In batch mode, if no tasks for a while, worker can exit
                if not self.streaming_mode and self.task_queue.empty():
                    break
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def reset(self):
        """Cancel worker and clear all queues."""
        self.cancel_flag.set()
        
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        self._clear_queue(self.task_queue)
        self._clear_queue(self.audio_queue)
        
        self.cancel_flag.clear()
        self.worker_thread = None
        self.task_counter = 0
        self.tasks_completed = 0
        self.streaming_mode = False
    
    def _clear_queue(self, q):
        """Clear a queue safely."""
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    
    def signal_streaming_complete(self):
        """Signal that streaming input is complete (called after flush)."""
        # Wait a bit for any final tasks to be queued
        import time
        time.sleep(0.1)
        self.streaming_mode = False
    
    def is_active(self) -> bool:
        """Check if worker is active or has pending tasks."""
        return (self.worker_thread and self.worker_thread.is_alive()) or not self.task_queue.empty() 