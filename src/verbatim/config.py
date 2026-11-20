"""Defaults, model ids and the optional user config file.

Precedence for every setting: command-line flag > config file > built-in default.
The config file lives at ~/.config/verbatim/config.toml (override with $VERBATIM_CONFIG).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

ASR_MODELS = {
    "vibevoice": "mlx-community/VibeVoice-ASR-bf16",  # diarized, up to ~60 min per pass
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v3",  # fastest, English-centric, no diarization
    "whisper": "mlx-community/whisper-large-v3-turbo",  # 99 languages, no diarization
}
DEFAULT_LLM = "mlx-community/Qwen3.6-35B-A3B-4bit"

SAMPLE_RATE = 16_000  # used for channel analysis (speaker labelling)
ANARLOG_SESSIONS_DIR = Path.home() / "Library/Application Support/anarlog/sessions"


def _default_output_dir() -> Path:
    """`output/` next to the source checkout (editable install), else ~/Documents/verbatim."""
    repo = Path(__file__).resolve().parents[2]
    if (repo / "pyproject.toml").exists():
        return repo / "output"
    return Path.home() / "Documents/verbatim"


def config_path() -> Path:
    return Path(os.environ.get("VERBATIM_CONFIG", "~/.config/verbatim/config.toml")).expanduser()


@dataclass
class Settings:
    """User-level defaults. Every field can be set in config.toml under the same name."""

    name: str = "Me"  # your name: used as your transcript label and in summaries
    hotwords: str = ""  # always-on names/jargon, merged with --hotwords
    engine: str = "vibevoice"
    llm: str = DEFAULT_LLM
    mic_channel: int = 1  # Anarlog: channel 1 = your mic, channel 2 = call audio
    sessions_dir: Path = field(default_factory=lambda: ANARLOG_SESSIONS_DIR)
    output_dir: Path = field(default_factory=_default_output_dir)

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or config_path()
        if not path.exists():
            return cls()
        data = tomllib.loads(path.read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown setting(s) {sorted(unknown)}; allowed: {sorted(known)}")
        for key in ("sessions_dir", "output_dir"):
            if key in data:
                data[key] = Path(data[key]).expanduser()
        if data.get("engine", "vibevoice") not in ASR_MODELS:
            raise ValueError(f"{path}: engine must be one of {list(ASR_MODELS)}")
        return cls(**data)
