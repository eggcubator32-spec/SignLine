"""Platform-specific speech-to-text services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class STTResult:
    """One speech-to-text result emitted by an STT backend."""

    text: str
    is_final: bool
    confidence: float = 1.0


class STTService(ABC):
    """Abstract speech-to-text interface."""

    @abstractmethod
    def start(self, on_result: Callable[[STTResult], None]) -> None:
        """Start listening and emit recognition results."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening and release recognizer resources."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return whether this STT backend can run on the current platform."""


class AndroidRecorderService(STTService):
    """Record and stream audio on Android using flet-audio-recorder."""

    def __init__(self, page) -> None:
        self._page = page
        self._recorder = None
        self._on_result: Callable[[STTResult], None] | None = None
        self._chunks: list[bytes] = []
        self._streaming = False

    def is_available(self) -> bool:
        """Return whether flet-audio-recorder is importable."""

        try:
            from flet_audio_recorder import AudioRecorder  # noqa: F401

            return True
        except ImportError:
            return False

    def start(self, on_result: Callable[[STTResult], None]) -> None:
        """Start recording and collect base64 PCM16 stream chunks."""

        try:
            import base64
            from flet_audio_recorder import AudioEncoder, AudioRecorder

            self._on_result = on_result
            self._chunks = []
            self._streaming = True

            def _handle_stream(event) -> None:
                if not self._streaming or not getattr(event, "data", None):
                    return
                try:
                    self._chunks.append(base64.b64decode(event.data))
                except Exception as exc:
                    print(f"[AndroidRecorder] chunk decode failed: {exc}")

            self._recorder = AudioRecorder(
                audio_encoder=AudioEncoder.PCM_16BIT,
                sample_rate=16_000,
                num_channels=1,
                on_stream=_handle_stream,
            )
            self._page.overlay.append(self._recorder)
            self._page.update()
            self._recorder.start_recording()
        except ImportError as exc:
            print(f"[AndroidRecorder] flet-audio-recorder unavailable: {exc}")
        except Exception as exc:
            print(f"[AndroidRecorder] start failed: {exc}")

    def stop(self) -> None:
        """Stop recording and emit a summary result."""

        try:
            self._streaming = False
            if self._recorder:
                self._recorder.stop_recording()
                try:
                    self._page.overlay.remove(self._recorder)
                    self._page.update()
                except Exception:
                    pass
                self._recorder = None

            if self._chunks and self._on_result:
                total_bytes = sum(len(chunk) for chunk in self._chunks)
                duration = total_bytes / 2 / 16_000
                self._on_result(
                    STTResult(
                        text=(
                            f"[Recorded {duration:.1f}s of audio - "
                            "offline transcription not available on Android]"
                        ),
                        is_final=True,
                    )
                )
            self._chunks = []
        except Exception as exc:
            print(f"[AndroidRecorder] stop failed: {exc}")


class DesktopSTTService(STTService):
    """Uses faster-whisper on desktop via existing ASRService."""

    def __init__(self, asr_config) -> None:
        self._config = asr_config
        self._service = None
        self._on_result: Callable[[STTResult], None] | None = None
        self._recording = False
        self._chunks: list[bytes] = []
        self._capture = None

    def is_available(self) -> bool:
        """Return whether faster-whisper is importable."""

        try:
            from faster_whisper import WhisperModel  # noqa: F401

            return True
        except ImportError:
            return False

    def start(self, on_result: Callable[[STTResult], None]) -> None:
        """Start desktop capture and buffer chunks until stop."""

        from services.asr_service import ASRService
        from services.mic_capture import make_capture

        self._service = ASRService(self._config)
        self._on_result = on_result
        self._chunks = []
        self._recording = True
        self._capture = make_capture()
        self._capture.start(self._on_chunk)

    def _on_chunk(self, audio_bytes: bytes) -> None:
        """Collect desktop audio chunks while recording."""

        if self._recording:
            self._chunks.append(audio_bytes)

    def stop(self) -> None:
        """Stop capture and transcribe buffered audio in a daemon thread."""

        self._recording = False
        if self._capture:
            self._capture.stop()
            self._capture = None
        if self._chunks and self._service and self._on_result:
            import threading

            threading.Thread(target=self._transcribe, daemon=True).start()

    def _transcribe(self) -> None:
        """Run desktop transcription in the background."""

        try:
            combined = b"".join(self._chunks)
            self._chunks = []
            result = self._service.transcribe_bytes(combined)
            text = str(result.get("text", "")).strip()
            if text and self._on_result:
                self._on_result(STTResult(text=text, is_final=True))
        except Exception as exc:
            print(f"[DesktopSTT] transcribe failed: {exc}")


class NoOpSTTService(STTService):
    """Silent fallback when no STT engine is available."""

    def start(self, on_result) -> None:
        """Start no-op recognition."""

    def stop(self) -> None:
        """Stop no-op recognition."""

    def is_available(self) -> bool:
        """Return False because this is a fallback."""

        return False


def _is_android() -> bool:
    """Return True when running on Android."""

    try:
        from plyer.utils import platform as plyer_platform

        return str(plyer_platform).lower() == "android"
    except Exception:
        import sys

        return hasattr(sys, "getandroidapilevel")


def make_stt_service(page=None, asr_config=None) -> STTService:
    """Return the right STT backend for the current platform."""

    if _is_android():
        if page is not None:
            svc = AndroidRecorderService(page)
            if svc.is_available():
                return svc
        return NoOpSTTService()

    if asr_config is not None:
        svc = DesktopSTTService(asr_config)
        if svc.is_available():
            return svc

    return NoOpSTTService()


__all__ = [
    "STTResult",
    "STTService",
    "AndroidRecorderService",
    "DesktopSTTService",
    "NoOpSTTService",
    "make_stt_service",
]
