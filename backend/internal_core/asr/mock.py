from __future__ import annotations

from .base import ASRProvider


class MockASRProvider(ASRProvider):
    def __init__(self) -> None:
        self._counter = 0

    def transcribe_chunk(
        self, wav_path: str, language: str = "en", timeout_sec: int = 30
    ) -> str:
        self._counter += 1
        return f"(mock) simulated transcript for chunk {self._counter}."

    def name(self) -> str:
        return "mock"
