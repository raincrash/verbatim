"""Command line: `verbatim list`, `verbatim run`, `verbatim config`."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __doc__ as PACKAGE_DOC
from . import anarlog
from .audio import require_ffmpeg
from .config import ASR_MODELS, Settings, config_path
from .pipeline import RunOptions, is_done, output_folder, run
from .speakers import parse_renames
from .transcript import output_index


def _merge_hotwords(*parts: str) -> str:
    words = [w.strip() for p in parts for w in p.split(",") if w.strip()]
    return ", ".join(dict.fromkeys(words))


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="verbatim", description=PACKAGE_DOC, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--sessions-dir", type=Path, default=settings.sessions_dir, help="Anarlog sessions folder")
        p.add_argument("--output", type=Path, default=settings.output_dir, help="where results are written")

    p_list = sub.add_parser("list", help="show Anarlog sessions (✓ = summary exists)")
    p_list.add_argument("-n", type=int, default=0, metavar="N", help="only the N most recent")
    common(p_list)

    p_run = sub.add_parser(
        "run",
        help="transcribe + summarize sessions or audio files",
        description="With no targets, processes every session that has no summary yet.",
    )
    p_run.add_argument("targets", nargs="*", help="Anarlog session id prefixes and/or audio/video file paths")
    p_run.add_argument("--latest", type=int, metavar="N", help="only the N most recent sessions")
    p_run.add_argument(
        "--name",
        default=settings.name,
        help=f"your name: your speaker label and how summaries refer to you (now: {settings.name!r})",
    )
    p_run.add_argument("--hotwords", default="", help="comma-separated names/jargon to spell right (added to config)")
    p_run.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="'OLD=NEW'",
        help="rename a speaker label, e.g. --rename 'Speaker 1=Bob' (repeatable; works on "
        "existing transcripts without re-transcribing)",
    )
    p_run.add_argument(
        "--engine",
        choices=list(ASR_MODELS),
        default=settings.engine,
        help="vibevoice: most accurate + who-spoke-when; parakeet: fastest; whisper: 99 languages",
    )
    p_run.add_argument("--llm", default=settings.llm, help="mlx-lm model repo used for summaries")
    p_run.add_argument("--no-summary", action="store_true", help="transcripts only")
    p_run.add_argument("--no-speakers", action="store_true", help="skip speaker labelling")
    p_run.add_argument(
        "--mic-channel",
        type=int,
        choices=[1, 2],
        default=settings.mic_channel,
        help="which stereo channel is your mic (Anarlog: 1)",
    )
    p_run.add_argument("--them", default="Them", help="label for the call side when the engine has no diarization")
    p_run.add_argument("--force", action="store_true", help="redo transcripts even if they exist")
    p_run.add_argument("--debug", action="store_true", help="print full tracebacks on failure")
    common(p_run)

    sub.add_parser("config", help="show the config file location and effective settings")
    return ap


def cmd_list(args: argparse.Namespace) -> int:
    opts = RunOptions(output_dir=args.output)
    index = output_index(args.output)
    sessions = anarlog.discover(args.sessions_dir)
    for s in sessions[: args.n or None]:
        mark = "✓" if is_done(s, opts, index) else " "
        print(f"{mark} {s.id[:8]}  {s.date:%Y-%m-%d %H:%M}  {s.title}")
    if not sessions:
        print(f"No recordings found in {args.sessions_dir}")
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    require_ffmpeg()
    opts = RunOptions(
        output_dir=args.output,
        engine=args.engine,
        llm=args.llm,
        me=args.name,
        them=args.them,
        mic_channel=args.mic_channel,
        label_speakers=not args.no_speakers,
        hotwords=_merge_hotwords(settings.hotwords, args.hotwords),
        renames=parse_renames(args.rename),
        force=args.force,
        summarize=not args.no_summary,
        debug=args.debug,
    )
    sessions = anarlog.discover(args.sessions_dir) if args.sessions_dir.exists() else []
    if args.targets:
        sessions = anarlog.resolve(args.targets, sessions)
    elif not args.force:
        index = output_index(args.output)
        sessions = [s for s in sessions if not is_done(s, opts, index)]
    if args.latest:
        sessions = sessions[: args.latest]
    if not sessions:
        print("Nothing to do (everything is summarized; pass a session id to redo one).")
        return 0
    failed = run(sessions, opts)
    if len(sessions) == 1 and not failed:
        print(f"Done: {output_folder(sessions[0], opts)}")
    return 1 if failed else 0


def cmd_config(settings: Settings) -> int:
    path = config_path()
    print(f"config file: {path} ({'found' if path.exists() else 'not created yet; built-in defaults in use'})")
    for key, value in vars(settings).items():
        print(f"  {key} = {str(value)!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        settings = Settings.load()
    except ValueError as e:
        raise SystemExit(f"config error: {e}") from e
    args = build_parser(settings).parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "config":
        return cmd_config(settings)
    try:
        return cmd_run(args, settings)
    except ValueError as e:  # bad --rename etc.
        raise SystemExit(str(e)) from e
