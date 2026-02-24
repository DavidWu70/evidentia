from __future__ import annotations

from abc import ABC, abstractmethod


class ASRError(RuntimeError):
    def __init__(self, code: str, message: str, provider_name: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.provider_name = provider_name


class ASRProvider(ABC):
    @abstractmethod
    def transcribe_chunk(
        self, wav_path: str, language: str = "en", timeout_sec: int = 30
    ) -> str: ...

    @abstractmethod
    def name(self) -> str: ...
