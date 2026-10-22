"""Delete saved photos from scan runs, so data/runs/ doesn't fill the SD card.

    pantilt clean                     # list runs and their photos, delete nothing
    pantilt clean --all               # delete the photos from every run
    pantilt clean --keep 2            # keep the newest 2 runs' photos, delete the rest
    pantilt clean --run board-20261022-141500
    pantilt clean --all --everything  # remove whole run folders, CSVs included

By default only the images go. poses.csv / tiles.csv and run.json stay, since
they're tiny and hold every probability for re-thresholding later. It always
shows what it is about to delete and asks first, unless --yes.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

PHOTO_DIRS = ("photos", "hits", "tiles")      # board-scan, survey hits, survey --save-all
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}


def _started(run: Path) -> str:
    """Sort key: the YYYYMMDD-HHMMSS in the name (survey and board-scan both use it)."""
    m = re.search(r"\d{8}-\d{6}", run.name)
    return m.group(0) if m else f"{run.stat().st_mtime:020.6f}"


def find_runs(runs_dir: Path) -> list[Path]:
    """Run folders, oldest first."""
    if not runs_dir.is_dir():
        return []
    return sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: (_started(p), p.name))


def photos_in(run: Path) -> list[Path]:
    return sorted(f for d in PHOTO_DIRS if (run / d).is_dir()
                  for f in (run / d).iterdir() if f.is_file() and f.suffix.lower() in PHOTO_EXTS)


def _size(files) -> int:
    return sum(f.stat().st_size for f in files)


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def select(runs: list[Path], all_: bool, keep: int | None, names: list[str]) -> list[Path]:
    if names:
        by_name = {r.name: r for r in runs}
        missing = [n for n in names if Path(n).name not in by_name]
        if missing:
            raise ValueError(f"no such run: {', '.join(missing)}")
        return [by_name[Path(n).name] for n in names]
    if keep is not None:
        return runs[:-keep] if keep > 0 else list(runs)
    return list(runs) if all_ else []


def cmd_clean(args, cfg) -> int:
    runs_dir = Path(args.runs_dir)
    runs = find_runs(runs_dir)
    if not runs:
        print(f"no runs in {runs_dir}")
        return 0

    try:
        chosen = select(runs, args.all, args.keep, args.run or [])
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    for r in runs:
        files = photos_in(r)
        mark = "  <- delete" if r in chosen else ""
        print(f"  {r.name:<28} {len(files):>4} photos  {_mb(_size(files)):>9}{mark}")

    if not chosen:
        print("\nnothing selected - use --all, --keep N or --run NAME")
        return 0

    if args.everything:
        files = [f for r in chosen for f in r.rglob("*") if f.is_file()]
        what = f"{len(chosen)} whole run folder{'s' if len(chosen) != 1 else ''} ({_mb(_size(files))})"
    else:
        files = [f for r in chosen for f in photos_in(r)]
        if not files:
            print("\nthe selected runs have no photos left")
            return 0
        what = f"{len(files)} photo{'s' if len(files) != 1 else ''} ({_mb(_size(files))})"

    if not args.yes:
        try:
            answer = input(f"\nPermanently delete {what}? This cannot be undone. [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            print("nothing deleted")
            return 1

    if args.everything:
        for r in chosen:
            shutil.rmtree(r)
    else:
        for f in files:
            f.unlink()
        for r in chosen:  # drop the now-empty photo folders
            for d in PHOTO_DIRS:
                if (r / d).is_dir() and not any((r / d).iterdir()):
                    (r / d).rmdir()
    print(f"deleted {what}")
    return 0
