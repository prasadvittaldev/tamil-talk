"""Parler-TTS (indic-parler-tts) synth backend: serves the public
checkpoint's Tamil named speakers (Jaya, Kavitha) through the `synth`
contract tamil_talk/server.py expects.

Lazily imported (parler_tts + torch are GPU-only, heavy) so this module can
be imported on CPU/test hosts without erroring.
"""
import os

from tamil_talk.text_chunk import chunk_text
from tamil_talk.leveler import level_stream, level_buffered

# Tamil speakers documented on the ai4bharat/indic-parler-tts model card.
# "voice" from the WS request selects a description string; Jaya is the
# model card's recommended Tamil speaker.
VOICE_MAP = {
    "tamil_female": (
        "Jaya speaks with a clear, moderate-pitch voice at a normal pace, "
        "in a close-sounding recording with almost no background noise."
    ),
    "jaya": (
        "Jaya speaks with a clear, moderate-pitch voice at a normal pace, "
        "in a close-sounding recording with almost no background noise."
    ),
    "kavitha": (
        "Kavitha speaks with a clear voice at a normal pace, in a "
        "close-sounding recording with almost no background noise."
    ),
}
DEFAULT_DESCRIPTION = VOICE_MAP["tamil_female"]


def make_parler_synth(model_name: str = "ai4bharat/indic-parler-tts",
                      quantize: str = None):
    """quantize: None (bf16/fp32 default), "int8", or "4bit" (bitsandbytes)."""
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quant_config = None
    if quantize == "int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    kwargs = {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    model = ParlerTTSForConditionalGeneration.from_pretrained(model_name, **kwargs)
    if quant_config is None:
        model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sample_rate = model.config.sampling_rate

    def _gen_pcm(text, description, temperature, repetition_penalty, max_tokens):
        import numpy as np
        desc_ids = description_tokenizer(description, return_tensors="pt").to(device)
        prompt_ids = tokenizer(text, return_tensors="pt").to(device)
        with torch.inference_mode():
            generation = model.generate(
                input_ids=desc_ids.input_ids,
                attention_mask=desc_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-3),
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_tokens,
            )
        audio = generation.to(torch.float32).cpu().numpy().squeeze()
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16).tobytes()

    def _gen_pcm_chunks(text, description, temperature, repetition_penalty, max_tokens):
        pcm = _gen_pcm(text, description, temperature, repetition_penalty, max_tokens)
        chunk = 4096 * 2  # bytes, stream the decoded sentence
        for i in range(0, len(pcm), chunk):
            yield pcm[i:i + chunk]

    raw_mode = os.environ.get("TTS_RAW") == "1"  # debug: skip chunk_text + leveler

    def synth(text, voice, temperature, repetition_penalty, max_tokens, speed=1.0):
        description = VOICE_MAP.get(voice, DEFAULT_DESCRIPTION)
        if raw_mode:
            yield from _gen_pcm_chunks(text, description, temperature, repetition_penalty, max_tokens)
            return
        for sent in chunk_text(text):
            chunks = _gen_pcm_chunks(sent, description, temperature, repetition_penalty, max_tokens)
            if speed == 1.0:
                yield from level_stream(chunks, sample_rate=sample_rate)
            else:
                yield from level_buffered(chunks, speed=speed, sample_rate=sample_rate)

    synth.sample_rate = sample_rate
    return synth
