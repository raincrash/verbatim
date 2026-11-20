"""Transcribe and summarize Anarlog recordings fully locally on Apple Silicon.

Pipeline per recording:
  audio.mp3 -> speech-to-text + who-spoke-when (VibeVoice-ASR 9B via mlx-audio; Parakeet / Whisper optional)
            -> your speaker found from the stereo channels (ch1 = your mic, ch2 = call audio)
            -> meeting notes with a local LLM (Qwen3.6-35B-A3B 4-bit via mlx-lm)
            -> output/<date>_<title>/{summary.md, transcript.md, transcript.json}
"""

__version__ = "0.2.0"


def main() -> int:
    """Console-script entry point (kept here so older installs that call `verbatim:main` keep working)."""
    from .cli import main as cli_main

    return cli_main()
