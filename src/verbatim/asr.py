"""Speech-to-text engines. Heavy ML imports happen inside the functions so the CLI starts fast."""

from __future__ import annotations

import re
from pathlib import Path

from .audio import load_stereo
from .config import ASR_MODELS
from .transcript import Segment

VIBEVOICE_SR = 24_000
VIBEVOICE_CHUNK_MIN = 55.0  # the model handles ~60 min per pass; longer recordings are split
_NON_SPEECH = re.compile(r"[\[(][^\])]*[\])]")  # [Music], (Silence), ...


def transcribe(path: Path, engine: str, context: str = "") -> list[Segment]:
    """Transcribe a file. `context` (title, names, jargon) biases spelling where the engine supports it."""
    if engine == "vibevoice":
        return _vibevoice(path, context)
    if engine == "parakeet":
        return _parakeet(path)
    if engine == "whisper":
        return _whisper(path, context)
    raise ValueError(f"unknown engine {engine!r}; choose from {list(ASR_MODELS)}")


def _vibevoice(path: Path, context: str) -> list[Segment]:
    """Diarized ASR. Speaker ids restart per chunk, so later chunks get a `/pN` suffix."""
    import mlx.core as mx
    from mlx_audio.stt import load

    model = load(ASR_MODELS["vibevoice"])
    mono = load_stereo(path, VIBEVOICE_SR).mean(axis=1)
    step = int(VIBEVOICE_CHUNK_MIN * 60 * VIBEVOICE_SR)
    segments: list[Segment] = []
    try:
        for part, start in enumerate(range(0, len(mono), step)):
            offset = start / VIBEVOICE_SR
            result = model.generate(
                mx.array(mono[start : start + step]),
                sampling_rate=VIBEVOICE_SR,
                context=context or None,
                max_tokens=32768,
            )
            for s in result.segments or []:
                text = (s.get("text") or s.get("content") or "").strip()
                if not text or _NON_SPEECH.fullmatch(text):
                    continue
                speaker = str(s.get("speaker_id", s.get("speaker", "")))
                if part:
                    speaker = f"{speaker}/p{part + 1}"
                segments.append(Segment(float(s["start"]) + offset, float(s["end"]) + offset, text, speaker))
    finally:
        del model
        mx.clear_cache()  # free the ~18 GB before the summary model loads
    return segments


def _parakeet(path: Path) -> list[Segment]:
    from parakeet_mlx import from_pretrained

    model = from_pretrained(ASR_MODELS["parakeet"])
    result = model.transcribe(str(path), chunk_duration=120.0, overlap_duration=15.0)
    return [Segment(s.start, s.end, s.text.strip()) for s in result.sentences if s.text.strip()]


def _whisper(path: Path, context: str) -> list[Segment]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=ASR_MODELS["whisper"],
        initial_prompt=context or None,
        condition_on_previous_text=False,
    )
    return [Segment(s["start"], s["end"], s["text"].strip()) for s in result["segments"] if s["text"].strip()]
