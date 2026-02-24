from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from .base import ASRError, ASRProvider


def whisper_cpp_available(bin_path: str, model_path: str) -> Tuple[bool, str]:
    if not bin_path:
        return False, "missing SCRIBE_WHISPER_CPP_BIN"
    if not model_path:
        return False, "missing SCRIBE_WHISPER_CPP_MODEL"
    if not Path(bin_path).exists():
        return False, f"whisper-cli not found: {bin_path}"
    if not Path(model_path).exists():
        return False, f"model not found: {model_path}"
    return True, ""


def _with_dyld_paths(bin_path: str, env: Optional[dict[str, str]] = None) -> dict[str, str]:
    env_out = dict(os.environ) if env is None else dict(env)
    if not bin_path:
        return env_out
    try:
        build_dir = Path(bin_path).resolve().parents[1]
    except Exception:
        return env_out

    candidates = [
        build_dir / "src",
        build_dir / "ggml" / "src",
        build_dir / "ggml" / "src" / "ggml-blas",
        build_dir / "ggml" / "src" / "ggml-metal",
    ]
    new_paths = [str(p) for p in candidates if p.exists()]
    if not new_paths:
        return env_out

    existing = env_out.get("DYLD_LIBRARY_PATH", "")
    joined = os.pathsep.join(new_paths)
    env_out["DYLD_LIBRARY_PATH"] = (
        joined if not existing else f"{joined}{os.pathsep}{existing}"
    )
    return env_out


def whisper_cpp_sanity_check(
    bin_path: str,
    model_path: str,
    sample_path: Path,
    *,
    language: str = "en",
    timeout_sec: int = 30,
    no_gpu: bool = False,
) -> Tuple[bool, str]:
    ok, reason = whisper_cpp_available(bin_path, model_path)
    if not ok:
        print(reason)
        return False, reason
    if not sample_path.exists():
        msg = f"sample not found: {sample_path}"
        print(msg)
        return False, msg

    with tempfile.TemporaryDirectory(prefix="whisper_sanity_") as tmp_dir:
        out_prefix = Path(tmp_dir) / "whisper_sanity"
        cmd = [
            bin_path,
            "-m",
            model_path,
            "-f",
            str(sample_path),
            "-l",
            language,
            "-otxt",
            "-of",
            str(out_prefix),
        ]
        if no_gpu:
            cmd.insert(1, "-ng")
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_sec,
                env=_with_dyld_paths(bin_path),
            )
        except subprocess.TimeoutExpired:
            msg = "whisper sanity check timed out"
            print(msg)
            return False, msg
        except Exception as e:
            msg = str(e)
            print(msg)
            return False, msg

        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            print(stderr)
            return False, stderr
        return True, ""


class WhisperCppProvider(ASRProvider):
    def __init__(self, bin_path: str, model_path: str, no_gpu: bool = False):
        self._bin_path = bin_path
        self._model_path = model_path
        self._no_gpu = bool(no_gpu)
        self._runtime_no_gpu = bool(no_gpu)
        self.last_decoding_method: str = "whisper_cpp"

    def name(self) -> str:
        return "whisper_cpp"

    def transcribe_chunk(
        self, wav_path: str, language: str = "en", timeout_sec: int = 30
    ) -> str:
        if not self._bin_path or not Path(self._bin_path).exists():
            raise ASRError(
                "WHISPER_BIN_MISSING",
                "whisper.cpp binary is not configured (set SCRIBE_WHISPER_CPP_BIN to whisper-cli)",
                self.name(),
            )
        if not self._model_path or not Path(self._model_path).exists():
            raise ASRError(
                "WHISPER_MODEL_MISSING",
                "whisper.cpp model is missing. Download a small model, e.g.: "
                "`cd ../whisper.cpp && bash ./models/download-ggml-model.sh small.en`, "
                "then set SCRIBE_WHISPER_CPP_MODEL=../whisper.cpp/models/ggml-small.en.bin",
                self.name(),
            )

        # Capture stdout (no output files) and keep logs clean.
        base_cmd = [
            self._bin_path,
            "-m",
            self._model_path,
            "-f",
            wav_path,
            "-l",
            language,
            "--no-timestamps",
            "--no-prints",
        ]

        # Some macOS builds crash in Metal path on certain machines; retry once on CPU.
        attempt_no_gpu = [True] if self._runtime_no_gpu else [False, True]
        attempt_errors: list[str] = []
        saw_timeout = False
        saw_empty_output = False

        for use_no_gpu in attempt_no_gpu:
            cmd = list(base_cmd)
            mode = "cpu_no_gpu" if use_no_gpu else "gpu_default"
            if use_no_gpu:
                cmd.insert(1, "-ng")
            try:
                res = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_sec,
                    env=_with_dyld_paths(self._bin_path),
                )
            except subprocess.TimeoutExpired:
                saw_timeout = True
                attempt_errors.append(f"{mode}: timeout")
                if not use_no_gpu:
                    self._runtime_no_gpu = True
                continue
            except Exception as e:
                attempt_errors.append(f"{mode}: {e}")
                if not use_no_gpu:
                    self._runtime_no_gpu = True
                continue

            if res.returncode != 0:
                msg = (res.stderr or "").strip() or f"exit_code={res.returncode}"
                if len(msg) > 200:
                    msg = msg[:200] + "…"
                attempt_errors.append(f"{mode}: {msg}")
                if not use_no_gpu:
                    self._runtime_no_gpu = True
                continue

            text_out = " ".join((res.stdout or "").split()).strip()
            if not text_out:
                saw_empty_output = True
                attempt_errors.append(f"{mode}: empty output")
                if not use_no_gpu:
                    self._runtime_no_gpu = True
                continue

            self.last_decoding_method = "whisper_cpp"
            return text_out

        if saw_timeout:
            raise ASRError("WHISPER_TIMEOUT", "; ".join(attempt_errors) or "whisper.cpp timed out", self.name())
        if saw_empty_output:
            raise ASRError(
                "WHISPER_EMPTY_OUTPUT",
                "; ".join(attempt_errors) or "whisper.cpp returned empty output",
                self.name(),
            )
        raise ASRError("WHISPER_EXIT_NONZERO", "; ".join(attempt_errors) or "non-zero exit", self.name())
