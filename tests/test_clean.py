"""Tests for `pantilt clean` - only ever deletes what it listed, and asks first."""

import builtins

import pytest

from pantilt.cli import main


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "runs"
    for name, sub in (("board-20261020-100000", "photos"), ("board-20261021-100000", "photos"),
                      ("20261022-100000", "hits")):
        (d / name / sub).mkdir(parents=True)
        for i in range(3):
            (d / name / sub / f"{i}.jpg").write_bytes(b"x" * 1000)
        (d / name / sub / "notes.txt").write_text("not a photo")
        (d / name / "poses.csv").write_text("index\n0\n")
    return d


def photos(d):
    return sorted(p.relative_to(d).as_posix() for p in d.rglob("*.jpg"))


def run(d, *argv, answer=None, monkeypatch=None):
    if answer is not None:
        monkeypatch.setattr(builtins, "input", lambda prompt: answer)
    return main(["clean", "--runs-dir", str(d), *argv])


def test_list_only_deletes_nothing(runs):
    assert run(runs) == 0
    assert len(photos(runs)) == 9


def test_declining_deletes_nothing(runs, monkeypatch):
    assert run(runs, "--all", answer="n", monkeypatch=monkeypatch) == 1
    assert len(photos(runs)) == 9


def test_all_deletes_photos_but_keeps_csv(runs, monkeypatch):
    assert run(runs, "--all", answer="y", monkeypatch=monkeypatch) == 0
    assert photos(runs) == []
    assert len(list(runs.rglob("poses.csv"))) == 3
    assert len(list(runs.rglob("notes.txt"))) == 3   # only images are touched


def test_keep_newest(runs):
    assert run(runs, "--keep", "1", "--yes") == 0
    # Newest by timestamp is the survey run, whatever the name prefix.
    assert photos(runs) == [f"20261022-100000/hits/{i}.jpg" for i in range(3)]


def test_one_run_and_everything(runs):
    assert run(runs, "--run", "20261022-100000", "--everything", "--yes") == 0
    assert not (runs / "20261022-100000").exists()
    assert len(photos(runs)) == 6


def test_unknown_run_is_an_error(runs):
    assert run(runs, "--run", "nope", "--yes") == 1
    assert len(photos(runs)) == 9


def test_no_runs_dir(tmp_path):
    assert run(tmp_path / "missing", "--all", "--yes") == 0
