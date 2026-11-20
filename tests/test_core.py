"""Fast tests for the pure logic (no models, no ffmpeg, no Anarlog)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verbatim import anarlog
from verbatim.cli import _merge_hotwords, build_parser
from verbatim.config import Settings
from verbatim.pipeline import RunOptions, output_folder
from verbatim.speakers import apply_renames, label_from_channels, number_speakers, parse_renames
from verbatim.summary import build_prompt
from verbatim.transcript import Segment, Transcript, merge_turns, output_index, timestamp

SR = 1_000


def test_timestamp():
    assert timestamp(0) == "00:00"
    assert timestamp(75.9) == "01:15"
    assert timestamp(3725) == "1:02:05"


def test_merge_turns_joins_same_speaker_within_gap():
    segs = [Segment(0, 1, "a", "X"), Segment(1.5, 2, "b", "X"), Segment(5, 6, "c", "X"), Segment(6, 7, "d", "Y")]
    out = merge_turns(segs)
    assert [(t.text, t.speaker) for t in out] == [("a b", "X"), ("c", "X"), ("d", "Y")]
    assert segs[0].text == "a"  # input untouched


def test_parse_and_apply_renames():
    renames = parse_renames(["Speaker 1=Bob", " Me = Alice "])
    assert renames == {"Speaker 1": "Bob", "Me": "Alice"}
    segs = [Segment(0, 1, "hi", "Speaker 1"), Segment(1, 2, "yo", "Speaker 2")]
    assert apply_renames(segs, renames)
    assert [s.speaker for s in segs] == ["Bob", "Speaker 2"]
    assert not apply_renames(segs, {"Nobody": "X"})


@pytest.mark.parametrize("bad", ["Speaker 1", "=Bob", "Speaker 1="])
def test_parse_renames_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_renames([bad])


def _two_channel_call() -> np.ndarray:
    """10 s: you talk on the mic for 0-5 s, the other side talks on the call channel for 5-10 s."""
    rng = np.random.default_rng(0)
    stereo = rng.normal(0, 0.001, (10 * SR, 2)).astype(np.float32)
    stereo[: 5 * SR, 0] += rng.normal(0, 0.3, 5 * SR)
    stereo[5 * SR :, 1] += rng.normal(0, 0.3, 5 * SR)
    return stereo


def test_label_from_channels_with_diarized_ids():
    segs = [Segment(0, 2, "a", "0"), Segment(2, 5, "b", "0"), Segment(5, 10, "c", "1")]
    label_from_channels(segs, _two_channel_call(), SR, me="Alice", them="Them", mic_channel=0)
    assert [s.speaker for s in segs] == ["Alice", "Alice", "Speaker 1"]


def test_label_from_channels_without_ids():
    segs = [Segment(0, 5, "a"), Segment(5, 10, "b")]
    label_from_channels(segs, _two_channel_call(), SR, me="Alice", them="Them", mic_channel=0)
    assert [s.speaker for s in segs] == ["Alice", "Them"]


def test_number_speakers_in_order_of_appearance():
    segs = [Segment(0, 1, "a", "7"), Segment(1, 2, "b", "3"), Segment(2, 3, "c", "7"), Segment(3, 4, "d", "")]
    number_speakers(segs)
    assert [s.speaker for s in segs] == ["Speaker 1", "Speaker 2", "Speaker 1", ""]


def _transcript(sid="abc-123") -> Transcript:
    return Transcript(
        sid,
        "T",
        "2026-01-01T10:00:00+00:00",
        "vibevoice",
        "m",
        65.0,
        [Segment(0, 1, "hello", "Alice"), Segment(2, 3, "hi", "Speaker 1")],
    )


def test_transcript_roundtrip_and_markdown(tmp_path: Path):
    t = _transcript()
    t.save(tmp_path)
    loaded = Transcript.load(tmp_path)
    assert loaded == t
    assert loaded.speakers() == ["Alice", "Speaker 1"]
    assert "`[00:02]` **Speaker 1** hi" in loaded.markdown("when")
    assert loaded.plain_text().splitlines()[0] == "[00:00] Alice: hello"


def test_output_folder_found_by_id_after_rename(tmp_path: Path):
    renamed = tmp_path / "my-own-folder-name"
    renamed.mkdir()
    _transcript("abc-123").save(renamed)
    (tmp_path / "junk").mkdir()
    (tmp_path / "junk" / "transcript.json").write_text("not json")
    assert output_index(tmp_path) == {"abc-123": renamed}

    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"")
    opts = RunOptions(output_dir=tmp_path)
    known = anarlog.Session("abc-123", audio, "New title", "2026-01-01T10:00:00Z")
    new = anarlog.Session("zzz", audio, "Fresh: call!", "2026-01-01T10:00:00Z")
    assert output_folder(known, opts) == renamed
    assert output_folder(new, opts).name.endswith("_Fresh-call")


def test_resolve_targets(tmp_path: Path):
    audio = tmp_path / "x.m4a"
    audio.write_bytes(b"")
    sessions = [anarlog.Session("aaaa1111", audio), anarlog.Session("aaaa2222", audio), anarlog.Session("bbbb", audio)]
    assert [s.id for s in anarlog.resolve(["bbbb"], sessions)] == ["bbbb"]
    picked = anarlog.resolve([str(audio)], sessions)
    assert picked[0].id == "x" and not picked[0].from_anarlog
    with pytest.raises(SystemExit, match="several"):
        anarlog.resolve(["aaaa"], sessions)
    with pytest.raises(SystemExit, match="No session"):
        anarlog.resolve(["cccc"], sessions)


def test_participant_names_parsing(monkeypatch):
    reply = {"data": {"participants": [{"display_name": "Bob"}, {"display_name": ""}, "Carol"]}}
    monkeypatch.setattr(anarlog, "_anarlog_json", lambda *a: reply)
    assert anarlog.participant_names("x") == ["Bob", "Carol"]
    monkeypatch.setattr(anarlog, "_anarlog_json", lambda *a: None)
    assert anarlog.participant_names("x") == []


def test_settings_load(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    assert Settings.load(cfg) == Settings()
    cfg.write_text('name = "Alice"\nhotwords = "Acme"\noutput_dir = "~/notes"\n')
    s = Settings.load(cfg)
    assert (s.name, s.hotwords, s.output_dir) == ("Alice", "Acme", Path.home() / "notes")
    cfg.write_text('nmae = "typo"\n')
    with pytest.raises(ValueError, match="unknown"):
        Settings.load(cfg)


def test_cli_defaults_come_from_settings():
    args = build_parser(Settings(name="Alice")).parse_args(["run", "abc", "--rename", "Speaker 1=Bob"])
    assert (args.name, args.targets, args.rename) == ("Alice", ["abc"], ["Speaker 1=Bob"])
    assert _merge_hotwords("A, B", "B,C,") == "A, B, C"


def test_prompt_mentions_recorder_and_names():
    p = build_prompt("Call", "Mon", "[00:00] Alice: hi", ["Alice", "Speaker 1"], me="Alice", names_hint="Carol, Dave")
    assert '"Alice" is the person who recorded the call' in p
    assert "Carol, Dave" in p and "Speaker 1" in p
    p = build_prompt("Call", "Mon", "x", ["Speaker 1"], me="Alice")
    assert "do not guess" in p
