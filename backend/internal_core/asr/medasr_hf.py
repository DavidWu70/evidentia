from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import sys

from .base import ASRError, ASRProvider


@dataclass
class _MedASRRuntime:
    torch: object
    processor: object
    model: object
    device: str
    decoder: Optional[object] = None


class MedASRHFProvider(ASRProvider):
    def __init__(
        self,
        model_ref: str,
        device: Optional[str] = None,
        fp16: bool = False,
        sample_rate: int = 16000,
        *,
        use_lm: bool = False,
        lm_path: str = "",
        lm_alpha: float = 0.5,
        lm_beta: float = 1.0,
        lm_beam_width: int = 50,
        min_text_chars: int = 2,
    ):
        self._model_ref = model_ref
        self._device_override = device
        self._fp16 = fp16
        self._sr = sample_rate
        self._use_lm = use_lm
        self._lm_path = lm_path
        self._lm_alpha = lm_alpha
        self._lm_beta = lm_beta
        self._lm_beam_width = lm_beam_width
        self._min_text_chars = min_text_chars
        self._rt: Optional[_MedASRRuntime] = None
        self.last_decoding_method: str = "argmax"
        self.lm_enabled: bool = bool(use_lm)
        self.lm_loaded: bool = False
        self.lm_status: str = "disabled"

    def name(self) -> str:
        return "medasr_hf"

    def _pick_device(self, torch_mod) -> str:
        if self._device_override:
            return self._device_override
        if getattr(torch_mod.cuda, "is_available", lambda: False)():
            return "cuda"
        backends = getattr(torch_mod, "backends", None)
        mps = getattr(backends, "mps", None) if backends else None
        if mps is not None and getattr(mps, "is_available", lambda: False)():
            return "mps"
        return "cpu"

    def _inject_medasr_venv_site_packages(self) -> bool:
        """
        Best-effort dependency bridge:
        allow runtime to reuse MedASR/.venv packages (e.g., transformers) when
        current interpreter environment is missing them.
        """
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates: list[Path] = []

        model_path = Path(str(self._model_ref or "")).expanduser()
        if model_path.exists():
            candidates.append((model_path / ".venv" / "lib" / py_ver / "site-packages").resolve())

        # backend/internal_core/asr/medasr_hf.py -> ... -> evidentia
        project_root = Path(__file__).resolve().parents[3]
        candidates.append((project_root / "MedASR" / ".venv" / "lib" / py_ver / "site-packages").resolve())
        candidates.append((project_root.parent / "MedASR" / ".venv" / "lib" / py_ver / "site-packages").resolve())

        injected = False
        for path in candidates:
            path_str = str(path)
            if not path.exists() or path_str in sys.path:
                continue
            sys.path.insert(0, path_str)
            injected = True
        return injected

    def _ensure_loaded(self) -> _MedASRRuntime:
        if self._rt is not None:
            return self._rt

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCTC, AutoProcessor  # type: ignore
        except Exception as e:
            if self._inject_medasr_venv_site_packages():
                try:
                    import torch  # type: ignore
                    from transformers import AutoModelForCTC, AutoProcessor  # type: ignore
                except Exception as e2:
                    raise ASRError(
                        "MEDASR_NOT_CONFIGURED",
                        f"MedASR requires torch+transformers installed: {e2}",
                        self.name(),
                    )
            else:
                raise ASRError(
                    "MEDASR_NOT_CONFIGURED",
                    f"MedASR requires torch+transformers installed: {e}",
                    self.name(),
                )

        model_ref = self._model_ref
        local_files_only = bool(model_ref) and Path(model_ref).exists()
        if not model_ref:
            raise ASRError(
                "MEDASR_NOT_CONFIGURED",
                "MedASR model is not configured (set SCRIBE_MEDASR_MODEL).",
                self.name(),
            )

        device = self._pick_device(torch)
        try:
            processor = AutoProcessor.from_pretrained(
                model_ref, local_files_only=local_files_only
            )
            model_kwargs: dict[str, object] = {"local_files_only": local_files_only}
            if self._fp16 and device in {"cuda", "mps"}:
                model_kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCTC.from_pretrained(model_ref, **model_kwargs).to(device)
            model.eval()
        except Exception as e:
            raise ASRError(
                "MEDASR_LOAD_FAILED",
                f"Failed to load MedASR model/processor: {e}",
                self.name(),
            )

        decoder = None
        if not self._use_lm:
            self.lm_loaded = False
            self.lm_status = "disabled (set MEDASR_USE_LM=true)"
        else:
            # LM is requested; try to load it, but always keep argmax as fallback.
            if not self._lm_path:
                self.lm_loaded = False
                self.lm_status = "enabled but unavailable (LM path empty; fallback argmax)"
            else:
                lm_path = Path(self._lm_path)
                if not lm_path.exists():
                    self.lm_loaded = False
                    self.lm_status = "enabled but unavailable (LM missing; fallback argmax)"
                else:
                    try:
                        import pyctcdecode  # type: ignore  # noqa: F401
                    except Exception:
                        decoder = None
                        self.lm_loaded = False
                        self.lm_status = "enabled but unavailable (pyctcdecode missing; fallback argmax)"
                    else:
                        try:
                            decoder = self._build_decoder(processor, str(lm_path))
                        except Exception as e:
                            decoder = None
                            msg = str(e).replace("\n", " ").strip()
                            if len(msg) > 160:
                                msg = msg[:160] + "…"
                            self.lm_loaded = False
                            self.lm_status = (
                                f"enabled but unavailable (decoder build failed: {msg}; fallback argmax)"
                            )
                        else:
                            self.lm_loaded = True
                            self.lm_status = "enabled + loaded"

        self._rt = _MedASRRuntime(
            torch=torch, processor=processor, model=model, device=device, decoder=decoder
        )
        return self._rt

    def _build_decoder(self, processor: Any, lm_path: str) -> object:
        try:
            import kenlm  # type: ignore  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"kenlm import failed: {e}")

        from pyctcdecode import build_ctcdecoder  # type: ignore

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "get_vocab"):
            raise RuntimeError("processor.tokenizer.get_vocab() unavailable")
        vocab = tokenizer.get_vocab()
        if not isinstance(vocab, dict) or not vocab:
            raise RuntimeError("tokenizer vocab empty")

        max_id = max(int(i) for i in vocab.values())
        labels = [""] * (max_id + 1)
        for token, idx in vocab.items():
            labels[int(idx)] = str(token)

        # CTC blank: use pad_token_id when available; otherwise assume 0.
        blank_id = getattr(tokenizer, "pad_token_id", 0)
        if blank_id is None:
            blank_id = 0
        if 0 <= int(blank_id) < len(labels):
            labels[int(blank_id)] = ""

        # Sanity: pyctcdecode requires unique non-blank labels.
        non_blank = [l for l in labels if l != ""]
        if len(set(non_blank)) != len(non_blank):
            raise RuntimeError("labels contain duplicate entries after normalization")

        return build_ctcdecoder(labels, kenlm_model_path=lm_path)

    def _load_audio_mono_16k_float32(self, path: Path):
        # Prefer miniaudio because we already use it for MP3->WAV fallback.
        try:
            import miniaudio  # type: ignore

            decoded = miniaudio.decode_file(
                str(path),
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=1,
                sample_rate=self._sr,
            )
            samples = decoded.samples
            try:
                import numpy as np  # type: ignore

                return np.asarray(samples, dtype="float32")
            except Exception:
                return samples
        except Exception:
            pass

        # Fallback to soundfile for WAV if available.
        try:
            import soundfile as sf  # type: ignore

            speech, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if getattr(speech, "ndim", 1) > 1:
                speech = speech[:, 0]
            if int(sr) != int(self._sr):
                raise ValueError(
                    f"Audio sample rate is {sr}Hz; expected {self._sr}Hz after normalization."
                )
            return speech
        except Exception as e:
            raise ASRError(
                "MEDASR_AUDIO_LOAD_FAILED",
                f"Failed to load audio for MedASR: {e}",
                self.name(),
            )

    def transcribe_chunk(
        self, wav_path: str, language: str = "en", timeout_sec: int = 30
    ) -> str:
        # Note: HF inference isn't easily interrupted cross-platform; timeout_sec is accepted for API parity.
        _ = (language, timeout_sec)
        rt = self._ensure_loaded()
        path = Path(wav_path)
        if not path.exists():
            raise ASRError("MEDASR_AUDIO_MISSING", "Audio chunk missing", self.name())

        try:
            speech = self._load_audio_mono_16k_float32(path)
            inputs = rt.processor(
                speech,
                sampling_rate=self._sr,
                return_tensors="pt",
                padding=False,
            )
            torch_mod = rt.torch
            with torch_mod.inference_mode():
                model_inputs = {
                    k: (v.to(rt.device) if getattr(torch_mod, "is_tensor")(v) else v)
                    for k, v in dict(inputs).items()
                }
                logits = rt.model(**model_inputs).logits

            text = ""
            # Optional KenLM CTC decoding.
            if rt.decoder is not None:
                try:
                    log_probs = torch_mod.log_softmax(logits, dim=-1).to("cpu").numpy()
                    # shape: [B, T, V]
                    text = rt.decoder.decode(
                        log_probs[0],
                        beam_width=int(self._lm_beam_width),
                        alpha=float(self._lm_alpha),
                        beta=float(self._lm_beta),
                    )
                    self.last_decoding_method = "lm"
                except Exception:
                    text = ""

            if len((text or "").strip()) < self._min_text_chars:
                pred_ids = torch_mod.argmax(logits, dim=-1).to("cpu")
                text = rt.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
                self.last_decoding_method = "argmax"

            # Post-process: normalize common delimiters without risking duplicates in the decoder labels.
            text = (text or "").replace("|", " ").replace("▁", " ")
            return " ".join(text.split()).strip()
        except ASRError:
            raise
        except Exception as e:
            raise ASRError("MEDASR_INFER_FAILED", str(e), self.name())
