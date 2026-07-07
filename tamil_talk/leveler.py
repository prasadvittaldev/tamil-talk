"""Per-sentence loudness leveling, silence trimming, and optional time-stretch.

For the token-streaming TTS path. Two entry points, both per sentence:
- `level_stream`: low-latency path (speed==1.0). Look-ahead leveling + leading-
  silence trim immediately, trailing-silence trim via a small tail-hold.
- `level_buffered`: buffers the whole sentence (speed!=1.0) to time-stretch +
  exactly level + trim both edges.

Trimming the model's leading/trailing silence per sentence removes the dead air
between utterances (sentences butt together with a small natural pad).
"""
import numpy as np


def _gain_for(samples, target_rms, max_gain, voiced_thresh):
    voiced = samples[np.abs(samples) > voiced_thresh]
    if voiced.size == 0:
        return 1.0
    rms = float(np.sqrt(np.mean(voiced * voiced)))
    if rms < 1e-6:
        return 1.0
    return min(max_gain, target_rms / rms)


def _apply(samples, gain, peak_ceiling):
    y = np.clip(samples * gain, -peak_ceiling, peak_ceiling)
    return (y * 32767.0).astype(np.int16).tobytes()


def _voiced_bounds(samples, voiced_thresh):
    v = np.where(np.abs(samples) > voiced_thresh)[0]
    if v.size == 0:
        return None
    return int(v[0]), int(v[-1])


def _trim_edges(samples, voiced_thresh, pad):
    b = _voiced_bounds(samples, voiced_thresh)
    if b is None:
        return samples[:0]
    lo = max(0, b[0] - pad)
    hi = min(len(samples), b[1] + 1 + pad)
    return samples[lo:hi]


def time_stretch_pcm(pcm_bytes: bytes, speed: float, peak_ceiling: float = 0.97) -> bytes:
    """Pitch-preserving time-stretch of s16le PCM. speed>1 = faster/shorter,
    speed<1 = slower/longer, speed==1.0 = unchanged. Uses a phase vocoder."""
    if speed == 1.0 or not pcm_bytes:
        return pcm_bytes
    import librosa  # lazy: only needed when speed != 1.0
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    y = librosa.effects.time_stretch(x, rate=speed)
    y = np.clip(y, -peak_ceiling, peak_ceiling)
    return (y * 32767.0).astype(np.int16).tobytes()


def level_buffered(chunk_iter, speed: float = 1.0, target_rms: float = 0.10,
                   max_gain: float = 12.0, voiced_thresh: float = 0.02,
                   peak_ceiling: float = 0.97, chunk_bytes: int = 8192,
                   sample_rate: int = 24000, trim_pad_sec: float = 0.03):
    """Buffer a whole sentence: optional time-stretch, trim leading+trailing
    silence, level to target_rms with one gain, re-emit in chunk_bytes pieces."""
    pcm = b"".join(chunk_iter)
    if not pcm:
        return
    pcm = time_stretch_pcm(pcm, speed, peak_ceiling)
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    x = _trim_edges(x, voiced_thresh, int(trim_pad_sec * sample_rate))
    if x.size == 0:
        return
    gain = _gain_for(x, target_rms, max_gain, voiced_thresh)
    out = _apply(x, gain, peak_ceiling)
    for i in range(0, len(out), chunk_bytes):
        yield out[i:i + chunk_bytes]


def level_stream(chunk_iter, target_rms: float = 0.10, lookahead_sec: float = 0.4,
                 sample_rate: int = 24000, max_gain: float = 12.0,
                 voiced_thresh: float = 0.02, peak_ceiling: float = 0.97,
                 trim_pad_sec: float = 0.03, tail_hold_sec: float = 0.35):
    """Yield gain-leveled, edge-trimmed s16le PCM for ONE sentence's chunk stream.

    Leading silence is trimmed from the first (look-ahead) buffer; trailing
    silence is trimmed by holding the last `tail_hold_sec` and trimming it at end.
    """
    need = int(lookahead_sec * sample_rate)
    pad = int(trim_pad_sec * sample_rate)
    hold = int(tail_hold_sec * sample_rate)
    buf, have, gain = [], 0, None
    pending = np.zeros(0, dtype=np.float32)
    voiced_seen = False

    def drain():  # emit everything in `pending` except the last `hold` samples
        nonlocal pending
        if pending.size > hold:
            cut = pending.size - hold
            chunk = pending[:cut]
            pending = pending[cut:]
            return _apply(chunk, gain, peak_ceiling)
        return None

    for chunk in chunk_iter:
        x = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if gain is None:
            buf.append(x)
            have += x.size
            if have >= need:
                pending = np.concatenate(buf)
                buf = []
                gain = _gain_for(pending, target_rms, max_gain, voiced_thresh)
        else:
            pending = np.concatenate([pending, x])
        if gain is None:
            continue
        if not voiced_seen:                 # hold (emit nothing) until speech starts
            b = _voiced_bounds(pending, voiced_thresh)
            if b is None:
                if pending.size > need:     # cap buffered silence
                    pending = pending[-need:]
                continue
            voiced_seen = True
            pending = pending[max(0, b[0] - pad):]   # trim LEADING silence
        out = drain()
        if out:
            yield out

    if gain is None:                        # short sentence: never filled look-ahead
        if not buf:
            return
        pending = np.concatenate(buf)
        gain = _gain_for(pending, target_rms, max_gain, voiced_thresh)
    pending = _trim_edges(pending, voiced_thresh, pad)   # trim leading+trailing on tail
    if pending.size:
        yield _apply(pending, gain, peak_ceiling)
