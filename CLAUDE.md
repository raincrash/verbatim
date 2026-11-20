# verbatim — notes for coding agents

Local transcription + meeting notes for [Anarlog](https://anarlog.so) recordings (or any audio file) on Apple Silicon. Everything runs on the user's Mac with MLX; never upload audio, transcripts or summaries anywhere.

## When the user asks to "run verbatim" on a call

1. `verbatim list` — newest first; match the call by time/title. "The latest" = the top row.
2. `verbatim run <id-prefix> --hotwords "<every name/company/jargon the user mentions>"`.
   Roughly 5–7 min per 15 min of audio plus ~1 min for the summary: run it in the background.
3. Read `output/<folder>/transcript.md`, work out who each "Speaker N" is from context and from who the user said they spoke with, then relabel:
   `verbatim run <id> --rename "Speaker 1=Bob"` (seconds; rewrites transcript + summary). The user's own label is the `name` from their config (`verbatim config`).
4. Report the summary (TL;DR, decisions, action items) and the folder path. Flag names the speech recognition clearly misheard; if it's bad, rerun with `--force` and better `--hotwords`.

Never hand-edit `transcript.md`/`summary.md` to fix labels — use `--rename` so `transcript.json` (the source of truth) stays consistent.

## Project facts

- Global command: `uv tool install --editable . --force` (editable: code edits apply immediately; reinstall only when `pyproject.toml` entry points/deps change).
- User settings: `~/.config/verbatim/config.toml` (`verbatim config` prints them). Personal defaults (name, hotwords) belong there, never in the repo.
- Outputs: `output/` (gitignored, personal data). Folders are matched to sessions by the `id` in `transcript.json`, so folder names may differ from titles.
- Anarlog data is read-only: `~/Library/Application Support/anarlog/sessions/<id>/audio.mp3` and `anarlog --json meetings list|get`.
- Memory: the ASR model (~18 GB) and the LLM (~20 GB) run in separate phases on purpose so 48 GB Macs work — keep it that way.

## Working on the code

- Layout and module responsibilities: see README "Development".
- Before committing: `uv run ruff check src tests && uv run ruff format src tests && uv run pytest`.
- Tests must stay model-free and fast; heavy imports (`mlx*`, `parakeet_mlx`) belong inside functions.
- Keep README tables in sync with CLI flags.
- Use made-up names (Alice, Bob, Acme) in examples and tests — never real people or real session ids.
