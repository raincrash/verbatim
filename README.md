# verbatim

Local, speaker-labelled transcripts and meeting notes for [Anarlog](https://anarlog.so) recordings — or any audio/video file — on Apple Silicon. Nothing leaves the machine: speech recognition, speaker detection and summaries all run locally with [MLX](https://github.com/ml-explore/mlx).

```bash
verbatim list                                   # recordings, ✓ = already summarized
verbatim run --latest 1                         # transcribe + summarize the newest recording
verbatim run abc12345 --hotwords "Bob, Acme"    # a specific session, with names spelled right
verbatim run abc12345 --rename "Speaker 1=Bob"  # put a real name on a speaker (no re-transcribing)
verbatim run ~/Downloads/interview.m4a          # any audio/video file ffmpeg can read
```

Output goes to `output/<date>_<title>/`:

| File | What |
|---|---|
| `summary.md` | TL;DR, key points, decisions, action items, open questions |
| `transcript.md` | timestamped, speaker-labelled transcript |
| `transcript.json` | the same as data; the source of truth that `summary.md` is built from |

## Typical workflow after a call

1. `verbatim list` to find the session id (first 8 characters are enough).
2. `verbatim run <id> --hotwords "names, company, jargon"` — names in `--hotwords` are used both to bias speech recognition and to tell the summarizer how to spell them.
3. Open `transcript.md`, see who "Speaker 1", "Speaker 2" are, then
   `verbatim run <id> --rename "Speaker 1=Bob" --rename "Speaker 2=Carol"`.
   This relabels the existing transcript in seconds and rewrites the summary with the real names.
4. If a name is consistently misheard, add it to `--hotwords` and redo the transcript with `--force`.

## Setup (once per Mac)

Needs an Apple Silicon Mac with ≥48 GB RAM, [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
brew install ffmpeg uv
git clone https://github.com/raincrash/verbatim.git && cd verbatim
uv sync                                  # project venv (for development / tests)
uv tool install --editable . --force     # global `verbatim` command; code edits apply immediately
mkdir -p ~/.config/verbatim && echo 'name = "Alice"' > ~/.config/verbatim/config.toml   # your name
```

Models download to `~/.cache/huggingface` on first use (~37 GB total).
Anarlog is optional. If it's installed, its recordings are found automatically and its CLI (`anarlog`) is used read-only for titles and dates; without the CLI, sessions are still found, just titled "Untitled". Without Anarlog, pass file paths to `verbatim run`.

## Settings

`~/.config/verbatim/config.toml` (all optional; flags override it; `verbatim config` shows what's in effect):

```toml
name = "Alice"                    # your speaker label + how summaries refer to you
hotwords = "Acme, Kubernetes"     # always-on names/jargon, merged with --hotwords
engine = "vibevoice"              # vibevoice | parakeet | whisper
llm = "mlx-community/Qwen3.6-35B-A3B-4bit"
mic_channel = 1                   # Anarlog records your mic on channel 1
output_dir = "~/Documents/verbatim"
sessions_dir = "~/Library/Application Support/anarlog/sessions"
```

## All `run` options

| Flag | Meaning |
|---|---|
| `targets…` | session id prefixes and/or file paths; none = every session without a summary |
| `--latest N` | only the N newest |
| `--hotwords "A, B"` | names/jargon to spell right |
| `--rename "OLD=NEW"` | relabel a speaker; repeatable |
| `--name NAME` | your label (default from config) |
| `--engine` | `vibevoice` (default: most accurate, who-spoke-when), `parakeet` (fastest, no speakers), `whisper` (99 languages, no speakers) |
| `--no-summary` | transcripts only |
| `--no-speakers` | skip speaker labelling |
| `--force` | redo the transcript even if one exists (summaries are always regenerated for the sessions you run) |
| `--debug` | full tracebacks |

## How it works

| Step | Model | Notes |
|---|---|---|
| Speech-to-text (default) | [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR) 9B, MLX bf16 via `mlx-audio` | up to ~60 min per pass with full context; outputs who-spoke-when and timestamps; takes the title + hotwords as context |
| `--engine parakeet` | [`parakeet-tdt-0.6b-v3`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) | much faster, very good English; no diarization (you vs. "Them") |
| `--engine whisper` | [`whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | 99 languages / code-switching; no diarization |
| Summary | [`Qwen3.6-35B-A3B` 4-bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) via `mlx-lm` | MoE, ~3B active params, ~20 GB |

**Who is you?** Anarlog records your mic on channel 1 and the call audio on channel 2. The diarized speaker heard mostly on the mic gets your name; everyone else is "Speaker 1, 2, …" by talk time. Files without separate channels just get "Speaker 1, 2, …".

Recordings over 55 min are split into parts; diarization restarts per part, so speakers in later parts are suffixed `/p2`, `/p3` (you are still detected in every part).

Transcription runs for all selected sessions first, then summaries, so the two big models never share RAM. Outputs are matched to sessions by the id inside `transcript.json`, so renaming an output folder or a meeting title in Anarlog is safe.

## Development

```bash
uv run pytest            # fast tests, no models needed
uv run ruff check src tests && uv run ruff format src tests
uv run python -m verbatim --help
```

Code map (`src/verbatim/`):

| Module | Responsibility |
|---|---|
| `cli.py` | argument parsing, `list` / `run` / `config` commands |
| `config.py` | model ids, defaults, `config.toml` loading |
| `anarlog.py` | finding sessions + metadata via the `anarlog` CLI, resolving CLI targets |
| `audio.py` | ffmpeg decoding |
| `asr.py` | the three speech-to-text engines |
| `speakers.py` | mic-vs-call speaker labelling, numbering, `--rename` |
| `transcript.py` | `Segment`/`Transcript` data, Markdown rendering, output folder lookup |
| `summary.py` | the summary prompt and the mlx-lm wrapper |
| `pipeline.py` | the two-phase transcribe → summarize run |

## License

[MIT](LICENSE). Model weights are downloaded from Hugging Face under their own licenses.
