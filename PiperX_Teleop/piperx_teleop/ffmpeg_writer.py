"""Stream BGR frames to FFmpeg and check encoder completion before committing."""
from pathlib import Path
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np


class FFmpegWriter:
    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.width, self.height = size
        if any(d <= 0 or d % 2 for d in size):
            raise ValueError("MP4 frame dimensions must be positive even numbers")
        self.closed = False
        executable = get_ffmpeg_exe()
        # A file avoids a full stderr pipe blocking the encoder during recording.
        self.log = self.path.with_suffix(".ffmpeg.log").open("w+b")
        command = [
            executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pixel_format", "bgr24",
            "-video_size", f"{self.width}x{self.height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2",
            "-movflags", "+faststart", str(self.path.resolve()),
        ]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self.log, bufsize=0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self.log.close()
            raise

    def _error(self, message):
        self.log.seek(0, 2)
        self.log.seek(max(0, self.log.tell()-8192))
        detail = self.log.read().decode("utf-8", errors="replace").strip()
        return OSError(f"FFmpeg {self.path.name}: {message}" + (f"; {detail}" if detail else ""))

    def write(self, frame):
        if self.closed:
            raise OSError("FFmpeg writer is closed")
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError("FFmpeg requires a uint8 BGR frame matching the configured size")
        if self.process.poll() is not None:
            raise self._error(f"encoder exited with code {self.process.returncode}")
        data = memoryview(np.ascontiguousarray(frame)).cast("B")
        try:
            # Raw pipes may accept only part of a frame in a single write.
            while data:
                written = self.process.stdin.write(data)
                if not written:
                    raise OSError("Encoder pipe accepted no data")
                data = data[written:]
        except OSError as error:
            raise self._error(f"frame write failed: {error}") from error

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except OSError:
                pass  # Still reap the process and check its exit status.
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired as error:
                self.process.kill()
                self.process.wait()
                raise self._error("timed out finalizing MP4") from error
            if code:
                raise self._error(f"encoder exited with code {code}")
        finally:
            self.log.close()
