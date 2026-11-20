"""Meeting notes from a transcript with a local LLM (mlx-lm)."""

from __future__ import annotations

import re

PROMPT = """You are an expert note-taker. Below is an automatic transcript of a recorded meeting titled "{title}" ({date}).
{who}{names}(Speaker labels and words may contain transcription errors - infer from context.)

Write meeting notes in Markdown with exactly these sections:

## TL;DR
2-4 sentences.

## Key points
Bullets grouped by topic. Be specific: names, numbers, dates, decisions, opinions.

## Decisions
Bullets (or "None recorded").

## Action items
Checklist "- [ ] Owner: task (deadline if mentioned)" (or "None recorded").

## Open questions / follow-ups
Bullets (or "None").

Do not invent facts that are not in the transcript. Output only the notes.

TRANSCRIPT:
{transcript}"""


def build_prompt(title: str, when: str, transcript: str, speakers: list[str], me: str, names_hint: str = "") -> str:
    """Fill in PROMPT, telling the model who recorded the call and which names to spell correctly."""
    generic = [s for s in speakers if s.startswith("Speaker ")]
    if me in speakers:
        who = f'"{me}" is the person who recorded the call; the other labels are the other people on it. '
    else:
        who = "The person who recorded the call is not identified; do not guess who it is. "
    if generic:
        who += (
            'Labels like "Speaker 1" are unidentified people: call them that unless the conversation makes '
            "clear who they are. "
        )
    names = f"People and terms that may come up (use these spellings): {names_hint}. " if names_hint else ""
    return PROMPT.format(title=title, date=when, who=who + "\n", names=names, transcript=transcript)


class Summarizer:
    """Loads the model once; call it with a prompt to get Markdown notes."""

    def __init__(self, repo: str):
        from mlx_lm import load

        self.repo = repo
        self.model, self.tokenizer = load(repo)

    def __call__(self, prompt: str, max_tokens: int = 2500) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "user", "content": prompt}]
        try:  # Qwen3-style models: skip the thinking phase
            text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        out = generate(
            self.model, self.tokenizer, prompt=text, max_tokens=max_tokens, sampler=make_sampler(temp=0.3, top_p=0.9)
        )
        return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
