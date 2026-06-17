"""
video_stream.py
---------------
Threaded video capture.

WHY THIS EXISTS (this is the core throughput optimization in the project):
Reading a frame from a webcam or video file is an I/O operation. If the main
program reads a frame, then runs detection, then reads the next frame, the
camera sits idle while detection runs, and detection sits idle while the next
frame is being read. The two steps serialize.

This class moves frame-reading into a SEPARATE BACKGROUND THREAD that keeps
grabbing the newest frame continuously. The main thread just asks for "the
latest frame available" whenever it is ready to process one. Capture and
inference now OVERLAP -- this is what raises throughput in the benchmark.

WHY THREADS WORK HERE DESPITE THE GIL: cv2.VideoCapture.read() is a native
OpenCV call that releases the Python GIL while it blocks on I/O, so the main
thread can run YOLO inference (also largely native) concurrently. This is an
I/O-overlap win, not CPU parallelism -- multiprocessing is unnecessary.
"""

import logging
import threading
import time

import cv2

log = logging.getLogger(__name__)


class StreamState:
    """Why read() returned None, so callers can react appropriately."""
    RUNNING = "running"
    ENDED = "ended"        # video file finished normally
    FAILED = "failed"      # camera/stream dropped unexpectedly


class ThreadedVideoStream:
    def __init__(self, source=0, reconnect_attempts=3, reconnect_delay=1.0):
        # source = 0 means the default webcam. It can also be a file path
        # like "video.mp4" or an RTSP/HTTP stream URL.
        self.source = source
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        # Read one frame immediately so self.frame is never None at startup.
        self.grabbed, self.frame = self.capture.read()

        self.state = StreamState.RUNNING
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Launch the background thread that continuously reads frames."""
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _is_file(self):
        # A non-numeric source that isn't a stream URL is treated as a file.
        return isinstance(self.source, str) and not self.source.lower().startswith(
            ("rtsp://", "http://", "https://")
        )

    def _reconnect(self):
        """Try to reopen a dropped LIVE stream. Files are not reconnected."""
        for attempt in range(1, self.reconnect_attempts + 1):
            log.warning("Stream dropped; reconnect attempt %d/%d",
                        attempt, self.reconnect_attempts)
            time.sleep(self.reconnect_delay)
            self.capture.release()
            self.capture = cv2.VideoCapture(self.source)
            if self.capture.isOpened():
                grabbed, frame = self.capture.read()
                if grabbed:
                    with self.lock:
                        self.grabbed, self.frame = grabbed, frame
                    log.info("Reconnected to %s", self.source)
                    return True
        return False

    def _update(self):
        """Runs in the background: keep grabbing the newest frame."""
        while self.running:
            grabbed, frame = self.capture.read()
            if not grabbed:
                if self._is_file():
                    log.info("End of video file reached.")
                    self.state = StreamState.ENDED
                    self.running = False
                    break
                # Live stream: try to recover before giving up.
                if not self._reconnect():
                    log.error("Could not recover stream after %d attempts.",
                              self.reconnect_attempts)
                    self.state = StreamState.FAILED
                    self.running = False
                    break
                continue
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        """Return the most recent frame, or None if the stream has stopped."""
        if not self.running and self.state != StreamState.RUNNING:
            return None
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Shut down the thread and release the camera."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.capture.release()
