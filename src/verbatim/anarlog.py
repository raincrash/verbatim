"""Read-only access to Anarlog recordings: audio files on disk + metadata from the `anarlog` CLI."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Session:
    """One recording: an Anarlog session or any standalone audio/video file."""

    id: str
    audio: Path
    title: str = "Untitled"
    started_at: str | None = None  # ISO timestamp from Anarlog; falls back to the file's mtime
    from_anarlog: bool = True

    @property
    def date(self) -> datetime:
        if self.started_at:
            try:
                return datetime.fromisoformat(self.started_at.replace("Z", "+00:00")).astimezone()
            except ValueError:
                pass
        return datetime.fromtimestamp(self.audio.stat().st_mtime).astimezone()

    @property
    def slug(self) -> str:
        """Folder name for this session's outputs, e.g. `2026-11-20_1130_Catch-Up`."""
        title = re.sub(r"[^\w\- ]+", "", self.title).strip().replace(" ", "-")[:60] or self.id[:8]
        return f"{self.date:%Y-%m-%d_%H%M}_{title}"

    @classmethod
    def from_file(cls, path: Path) -> Session:
        return cls(id=path.stem, audio=path, title=path.stem, from_anarlog=False)


def _anarlog_exe() -> str:
    return shutil.which("anarlog") or str(Path.home() / ".local/bin/anarlog")


def _anarlog_json(*args: str) -> dict | None:
    """Run `anarlog --json …` and return the parsed reply, or None if the CLI is missing or fails."""
    try:
        r = subprocess.run([_anarlog_exe(), "--json", *args], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def meetings() -> dict[str, dict]:
    """All meetings Anarlog knows about, keyed by session id (titles, dates)."""
    out: dict[str, dict] = {}
    offset = 0
    while True:
        reply = _anarlog_json("meetings", "list", "--limit", "200", "--offset", str(offset))
        if not reply or not reply.get("data"):
            return out
        out.update({m["id"]: m for m in reply["data"]})
        offset = (reply.get("pagination") or {}).get("next_offset")
        if not offset:
            return out


def participant_names(session_id: str) -> list[str]:
    """Named participants of a meeting (often empty: Anarlog only knows names you've entered)."""
    reply = _anarlog_json("meetings", "get", session_id)
    if not reply:
        return []
    names = []
    for p in (reply.get("data") or {}).get("participants") or []:
        if isinstance(p, dict):
            name = p.get("display_name") or p.get("name") or p.get("full_name")
        else:
            name = str(p)
        if name:
            names.append(name)
    return names


def discover(sessions_dir: Path) -> list[Session]:
    """Every Anarlog session with audio, newest first."""
    meta = meetings()
    sessions = []
    for audio in sessions_dir.glob("*/audio.mp3"):
        sid = audio.parent.name
        m = meta.get(sid, {})
        sessions.append(Session(sid, audio, m.get("title") or "Untitled", m.get("started_at") or m.get("created_at")))
    return sorted(sessions, key=lambda s: s.date, reverse=True)


def resolve(targets: list[str], sessions: list[Session]) -> list[Session]:
    """Turn CLI targets (session-id prefixes or file paths) into sessions. Unknown targets raise."""
    by_id = {s.id: s for s in sessions}
    picked: list[Session] = []
    for target in targets:
        path = Path(target).expanduser()
        if path.is_file():
            path = path.resolve()
            known = by_id.get(path.parent.name)  # a file inside an Anarlog session folder keeps its title
            picked.append(known if known and known.audio == path else Session.from_file(path))
            continue
        matches = [s for s in sessions if s.id.startswith(target)]
        if not matches:
            raise SystemExit(f"No session or file matches {target!r} (see `verbatim list`).")
        if len(matches) > 1:
            ids = ", ".join(s.id[:8] for s in matches)
            raise SystemExit(f"{target!r} matches several sessions ({ids}); use a longer prefix.")
        picked += matches
    return picked
