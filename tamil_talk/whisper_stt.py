"""Tamil Whisper STT backend: wraps faster-whisper's
vasista22/whisper-tamil-large-v2 as a single make_whisper_stt(...) ->
transcribe(pcm, sample_rate) callable.

faster_whisper is imported lazily inside make_whisper_stt so this module can
be imported on hosts without the package/GPU.
"""
import logging
import os

import numpy as np

from tamil_talk.resampler import PcmResampler

logger = logging.getLogger(__name__)

CT2_CACHE_DIR = os.path.expanduser("~/.cache/ct2-models")


def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _resample_to_16k(pcm_bytes: bytes, src_rate: int) -> bytes:
    if src_rate == 16000:
        return pcm_bytes
    r = PcmResampler(src_rate, 16000)
    return r.process(pcm_bytes) + r.flush()


def make_whisper_stt(model_name: str = "vasista22/whisper-tamil-large-v2",
                     device: str = "cuda", compute_type: str = "float16"):
    """Create a transcribe callable using faster-whisper (Tamil-tuned model).

    Args:
        model_name: A CTranslate2-converted Whisper model, a standard Whisper
            size, or (the default) `vasista22/whisper-tamil-large-v2` — a raw
            HF Transformers checkpoint. faster_whisper's WhisperModel only
            accepts already-CT2-converted repos, so if loading it directly
            raises RuntimeError we convert it once via ctranslate2's
            TransformersConverter, cache the result under CT2_CACHE_DIR, and
            load from the cached directory on this and subsequent calls.
        device: "cuda" or "cpu"
        compute_type: "float16", "int8", etc. for CTranslate2 quantization

    Returns:
        A callable transcribe(pcm_bytes: bytes, sample_rate: int) -> str
    """
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except RuntimeError as e:
        if "model.bin" not in str(e):
            raise
        # Not a pre-converted CTranslate2 repo (e.g. a raw HF Transformers
        # checkpoint) -- convert once, cache to disk, load from there.
        from ctranslate2.converters import TransformersConverter

        class _CompatConverter(TransformersConverter):
            # ctranslate2 4.8.0 unconditionally passes `dtype=` to
            # `from_pretrained`, but transformers<5 only recognizes/pops
            # `torch_dtype`, so `dtype` leaks into the model constructor and
            # raises TypeError. Translate it before delegating.
            def load_model(self, model_class, model_name_or_path, **kwargs):
                dtype = kwargs.pop("dtype", None)
                if dtype is not None:
                    kwargs["torch_dtype"] = dtype
                return model_class.from_pretrained(model_name_or_path, **kwargs)

        out_dir = os.path.join(CT2_CACHE_DIR, model_name.replace("/", "--"))
        if not os.path.exists(os.path.join(out_dir, "model.bin")):
            logger.info(
                f"Converting {model_name} to CTranslate2 format (one-time, "
                f"several GB, cached under {CT2_CACHE_DIR})..."
            )
            _CompatConverter(model_name).convert(out_dir, force=True)
        model = WhisperModel(out_dir, device=device, compute_type=compute_type)

    def transcribe(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        pcm16k = _resample_to_16k(pcm_bytes, sample_rate)
        audio = pcm_bytes_to_float32(pcm16k)
        segments, _ = model.transcribe(audio, language="ta")
        return " ".join(seg.text.strip() for seg in segments).strip()

    return transcribe
