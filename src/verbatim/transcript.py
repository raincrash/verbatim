"""Transcript data model, formatting and on-disk layout.

Each session gets `output/<slug>/` with transcript.json (source of truth), transcript.md and summary.md.
The session id is stored in transcript.json, so outputs are found even if the folder is renamed
or the meeting title changes in Anarlog.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

TRANSCRIPT_JSON = "transcript.json"
TRANSCRIPT_MD = "transcript.md"
SUMMARY_MD = "summary.md"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""


@dataclass
class Transcript:
    id: str
    title: str
    date: str  # ISO timestamp
    engine: str
    model: str
    duration_sec: float
    turns: list[Segment]

    def save(self, folder: Path) -> None:
        data = asdict(self)
        (folder / TRANSCRIPT_JSON).write_text(json.dumps(data, indent=1, ensure_ascii=False))

    @classmethod
    def load(cls, folder: Path) -> Transcript:
        data = json.loads((folder / TRANSCRIPT_JSON).read_text())
        data["turns"] = [Segment(**t) for t in data["turns"]]
        return cls(**data)

    def speakers(self) -> list[str]:
        return list(dict.fromkeys(t.speaker for t in self.turns if t.speaker))

    def plain_text(self) -> str:
        """Compact `[mm:ss] Speaker: text` lines, used as LLM input."""
        return "\n".join(f"[{timestamp(t.start)}] {t.speaker + ': ' if t.speaker else ''}{t.text}" for t in self.turns)

    def markdown(self, when: str) -> str:
        header = f"*{when} · {timestamp(self.duration_sec)} · transcribed locally with `{self.model}`*"
        lines = [f"# {self.title}", header, ""]
        for t in self.turns:
            who = f"**{t.speaker}** " if t.speaker else ""
            lines.append(f"`[{timestamp(t.start)}]` {who}{t.text}\n")
        return "\n".join(lines)


def timestamp(sec: float) -> str:
    """`mm:ss`, or `h:mm:ss` from one hour."""
    sec = int(sec)
    if sec >= 3600:
        return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
    return f"{sec // 60:02d}:{sec % 60:02d}"


def merge_turns(segments: list[Segment], gap: float = 2.0) -> list[Segment]:
    """Join consecutive segments by the same speaker that are less than `gap` seconds apart."""
    out: list[Segment] = []
    for s in segments:
        if out and out[-1].speaker == s.speaker and s.start - out[-1].end < gap:
            out[-1].end, out[-1].text = s.end, f"{out[-1].text} {s.text}"
        else:
            out.append(Segment(s.start, s.end, s.text, s.speaker))
    return out


def output_index(output_dir: Path) -> dict[str, Path]:
    """Map session id -> existing output folder, read from each folder's transcript.json."""
    index: dict[str, Path] = {}
    for tj in output_dir.glob(f"*/{TRANSCRIPT_JSON}"):
        try:
            index[json.loads(tj.read_text())["id"]] = tj.parent
        except (OSError, ValueError, KeyError):
            continue
    return index
