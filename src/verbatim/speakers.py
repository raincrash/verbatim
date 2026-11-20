"""Speaker labels: find which diarized speaker is you, number the rest, apply manual renames."""

from __future__ import annotations

import numpy as np

from .transcript import Segment


def _db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9))


def label_from_channels(
    segments: list[Segment], stereo: np.ndarray, sample_rate: int, me: str, them: str, mic_channel: int
) -> None:
    """Label speakers in place using Anarlog's two channels (your mic vs call audio).

    Per segment, whichever channel is louder relative to its own typical level decides mic vs call.
    With diarized ids, the id mostly heard on the mic becomes `me` and the others "Speaker 1, 2, …"
    (ordered by talk time). Without ids, each segment is simply `me` or `them`.
    `mic_channel` is 0-based.
    """
    mic, call = stereo[:, mic_channel], stereo[:, 1 - mic_channel]
    win = sample_rate // 2
    n = len(mic) // win
    if not n or not segments:
        return

    def ref_level(ch: np.ndarray) -> float:  # typical level while that channel is active
        levels = np.array([_db(ch[i * win : (i + 1) * win]) for i in range(n)])
        return float(np.median(levels[levels >= np.percentile(levels, 60)]))

    mic_ref, call_ref = ref_level(mic), ref_level(call)

    def on_mic(s: Segment) -> bool:
        a = int(s.start * sample_rate)
        b = max(int(s.end * sample_rate), a + 1)
        return (_db(mic[a:b]) - mic_ref) > (_db(call[a:b]) - call_ref)

    if not any(s.speaker for s in segments):
        for s in segments:
            s.speaker = me if on_mic(s) else them
        return

    seconds: dict[str, list[float]] = {}  # speaker id -> [seconds on mic, seconds on call]
    for s in segments:
        seconds.setdefault(s.speaker, [0.0, 0.0])[0 if on_mic(s) else 1] += max(s.end - s.start, 0.1)
    names: dict[str, str] = {}
    k = 1
    for spk, (on_mic_s, on_call_s) in sorted(seconds.items(), key=lambda kv: -sum(kv[1])):
        if on_mic_s > on_call_s:
            names[spk] = me
        else:
            names[spk] = f"Speaker {k}"
            k += 1
    for s in segments:
        s.speaker = names[s.speaker]


def number_speakers(segments: list[Segment]) -> None:
    """For files without mic/call channels: map raw diarization ids to "Speaker 1, 2, …" in order of appearance."""
    ids: dict[str, str] = {}
    for s in segments:
        if s.speaker:
            s.speaker = ids.setdefault(s.speaker, f"Speaker {len(ids) + 1}")


def parse_renames(pairs: list[str]) -> dict[str, str]:
    """`["Speaker 1=Bob", "Speaker 2 = Carol"]` -> {"Speaker 1": "Bob", "Speaker 2": "Carol"}."""
    out: dict[str, str] = {}
    for pair in pairs:
        old, sep, new = pair.partition("=")
        if not sep or not old.strip() or not new.strip():
            raise ValueError(f"bad --rename {pair!r}; expected 'Old label=New name'")
        out[old.strip()] = new.strip()
    return out


def apply_renames(segments: list[Segment], renames: dict[str, str]) -> bool:
    """Rename speaker labels in place. Returns True if anything changed."""
    changed = False
    for s in segments:
        if s.speaker in renames:
            s.speaker = renames[s.speaker]
            changed = True
    return changed
