"""The two-phase pipeline: transcribe every session, then summarize every session.

The phases are separate so the ASR model (~18 GB) and the LLM (~20 GB) are never in memory together.
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import anarlog
from .asr import transcribe
from .audio import has_distinct_channels, load_stereo
from .config import ASR_MODELS, SAMPLE_RATE
from .speakers import apply_renames, label_from_channels, number_speakers
from .summary import Summarizer, build_prompt
from .transcript import SUMMARY_MD, TRANSCRIPT_JSON, TRANSCRIPT_MD, Transcript, merge_turns, output_index, timestamp


@dataclass
class RunOptions:
    output_dir: Path
    engine: str = "vibevoice"
    llm: str = ""
    me: str = "Me"  # your label in transcripts and summaries
    them: str = "Them"  # label for the call side when the engine has no diarization
    mic_channel: int = 1  # 1-based, as on the command line
    label_speakers: bool = True
    hotwords: str = ""
    renames: dict[str, str] = field(default_factory=dict)
    force: bool = False
    summarize: bool = True
    debug: bool = False


def output_folder(session: anarlog.Session, opts: RunOptions, index: dict[str, Path] | None = None) -> Path:
    """The existing output folder for this session (found by id), else a new one named by date + title."""
    index = output_index(opts.output_dir) if index is None else index
    return index.get(session.id) or opts.output_dir / session.slug


def is_done(session: anarlog.Session, opts: RunOptions, index: dict[str, Path]) -> bool:
    return (output_folder(session, opts, index) / SUMMARY_MD).exists()


def _when(session: anarlog.Session) -> str:
    return f"{session.date:%a %d %b %Y, %H:%M}"


def _write_transcript(transcript: Transcript, session: anarlog.Session, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    transcript.save(folder)
    (folder / TRANSCRIPT_MD).write_text(transcript.markdown(_when(session)))


def transcribe_session(session: anarlog.Session, opts: RunOptions, folder: Path) -> Transcript:
    """Phase 1. Reuses an existing transcript unless --force; manual renames are applied either way."""
    if (folder / TRANSCRIPT_JSON).exists() and not opts.force:
        transcript = Transcript.load(folder)
        if apply_renames(transcript.turns, opts.renames):
            _write_transcript(transcript, session, folder)
            print(f"  renamed speakers in existing transcript -> {', '.join(transcript.speakers())}")
        else:
            print(f"  transcript exists ({len(transcript.turns)} turns), skipping ASR (use --force to redo)")
        return transcript

    t0 = time.time()
    names = anarlog.participant_names(session.id) if session.from_anarlog else []
    title = session.title if session.title != "Untitled" else ""
    context = "; ".join(x for x in [title, ", ".join(names), opts.hotwords] if x)
    segments = transcribe(session.audio, opts.engine, context)

    stereo = load_stereo(session.audio, SAMPLE_RATE)
    duration = len(stereo) / SAMPLE_RATE
    if opts.label_speakers and has_distinct_channels(stereo):
        label_from_channels(segments, stereo, SAMPLE_RATE, opts.me, opts.them, opts.mic_channel - 1)
    else:  # mono / non-Anarlog file: we can't tell which speaker is you
        number_speakers(segments)
    apply_renames(segments, opts.renames)

    transcript = Transcript(
        id=session.id,
        title=session.title,
        date=session.date.isoformat(),
        engine=opts.engine,
        model=ASR_MODELS[opts.engine],
        duration_sec=duration,
        turns=merge_turns(segments),
    )
    _write_transcript(transcript, session, folder)
    print(
        f"  transcribed {timestamp(duration)} of audio in {time.time() - t0:.0f}s "
        f"({len(segments)} sentences; speakers: {', '.join(transcript.speakers()) or 'none'})"
    )
    return transcript


def summarize_session(session: anarlog.Session, opts: RunOptions, folder: Path, summarizer: Summarizer) -> None:
    """Phase 2. Always (re)writes summary.md from the transcript on disk."""
    t0 = time.time()
    transcript = Transcript.load(folder)
    prompt = build_prompt(
        session.title,
        _when(session),
        transcript.plain_text(),
        transcript.speakers(),
        me=opts.me,
        names_hint=opts.hotwords,
    )
    notes = summarizer(prompt)
    (folder / SUMMARY_MD).write_text(
        f"# {session.title}\n*{_when(session)} · summarized locally with `{summarizer.repo}`*\n\n{notes}\n\n"
        f"---\nFull transcript: [{TRANSCRIPT_MD}]({TRANSCRIPT_MD})\n"
    )
    print(f"  summarized in {time.time() - t0:.0f}s -> {folder / SUMMARY_MD}")


def _report(e: Exception, opts: RunOptions) -> None:
    if opts.debug:
        traceback.print_exc()
    else:
        print(f"  !! failed: {e!r} (rerun with --debug for the traceback)", file=sys.stderr)


def run(sessions: list[anarlog.Session], opts: RunOptions) -> int:
    """Process sessions; returns the number that failed (keeps going past bad files)."""
    index = output_index(opts.output_dir)
    folders = {s.id: output_folder(s, opts, index) for s in sessions}
    failed: set[str] = set()

    for i, s in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] transcribing: {s.title}  ({s.date:%Y-%m-%d %H:%M}, {s.id[:8]})")
        try:
            transcribe_session(s, opts, folders[s.id])
            print(f"  -> {folders[s.id]}")
        except Exception as e:
            failed.add(s.id)
            _report(e, opts)

    todo = [s for s in sessions if s.id not in failed and (folders[s.id] / TRANSCRIPT_JSON).exists()]
    if opts.summarize and todo:
        print(f"Loading {opts.llm} (the first run downloads ~20 GB)...")
        summarizer = Summarizer(opts.llm)
        for i, s in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] summarizing: {s.title}")
            try:
                summarize_session(s, opts, folders[s.id], summarizer)
            except Exception as e:
                failed.add(s.id)
                _report(e, opts)
    return len(failed)
