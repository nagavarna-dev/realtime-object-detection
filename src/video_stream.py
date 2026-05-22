"""
video_stream.py
---------------
Threaded video capture.

WHY THIS EXISTS (this is the core latency optimization in the project):
Reading a frame from a webcam or video file is an I/O operation. If the main
program reads a frame, then runs detection, then reads the next frame, the
camera sits idle while detection runs, and detection sits idle while the next
frame is being read. The two steps block each other.

This class moves frame-reading into a SEPARATE BACKGROUND THREAD that keeps
grabbing the newest frame continuously. The main thread just asks for "the
latest frame available" whenever it is ready to process one. Capture and
inference now overlap instead of waiting for each other -- this is what
produces the latency improvement reported in the benchmark.
"""

import threading
import cv2


class ThreadedVideoStream:
    def __init__(self, source=0):
        # source = 0 means the default webcam. It can also be a file path
        # like "video.mp4" or an RTSP/HTTP stream URL.
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        # Read one frame immediately so self.frame is never None at startup.
        self.grabbed, self.frame = self.capture.read()

        # 'running' controls the background loop; 'lock' protects the shared
        # frame so the reader thread and main thread don't clash.
        self.running = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Launch the background thread that continuously reads frames."""
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """Runs in the background: keep grabbing the newest frame forever."""
        while self.running:
            grabbed, frame = self.capture.read()
            if not grabbed:
                # End of a video file, or camera dropped out.
                self.running = False
                break
            # Replace the stored frame with the freshest one.
            with self.lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        """Return the most recent frame (called by the main thread)."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Shut down the thread and release the camera."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.capture.release()
