"""Stateful streaming PCM resampler (s16le mono) via soxr.

SNAC/synth output is 24 kHz; Pipecat's Azure-style contract wants 16 kHz
(Raw16Khz16BitMonoPcm). One instance per connection keeps filter state across
chunks so there are no boundary clicks.
"""
import numpy as np
import soxr


class PcmResampler:
    def __init__(self, src_rate: int = 24000, dst_rate: int = 16000):
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._passthrough = src_rate == dst_rate
        self._stream = None if self._passthrough else \
            soxr.ResampleStream(src_rate, dst_rate, 1, dtype="int16")

    def process(self, pcm: bytes) -> bytes:
        if len(pcm) % 2:
            pcm = pcm[:-1]  # drop stray byte so int16 framing stays aligned
        if self._passthrough:
            return pcm
        arr = np.frombuffer(pcm, dtype=np.int16)
        out = self._stream.resample_chunk(arr)
        return out.tobytes()

    def flush(self) -> bytes:
        if self._passthrough:
            return b""
        out = self._stream.resample_chunk(np.zeros(0, dtype=np.int16), last=True)
        return out.tobytes()
