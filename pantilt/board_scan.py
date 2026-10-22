"""Scan only the board: teach its corners once, then hunt it for craters.

`pantilt live --crop 0.2` classifies whatever the crosshair is on. This walks
the crosshair over a grid that stays inside the board, stops at each point,
waits for the image to stop shaking, and saves a photo when the target class
(crater by default) clears a confidence threshold.

    pantilt teach                         # aim at the 4 board corners, save them
    pantilt board-scan --crop 0.2 --dry-run   # print the grid, move nothing
    pantilt board-scan --crop 0.2 --trace     # visit the grid's outer points, check the red box
    pantilt board-scan --crop 0.2 --threshold 0.7

Why corners instead of measuring: the taught angles already include where the
camera sits, how the board is tilted, and which way the servos turn, so there
is nothing to measure and no sign to get wrong. The grid is inset by half a
crop square, so the patch the model sees never hangs off the board edge.

Why stop-and-go: a frame judged while the mount is moving is blurred, and a
blurred crater is not a crater to the model. Each pose waits `settle_s`, then
keeps waiting (up to `settle_max_s`) until two frames in a row look the same,
and only then lets the classifier's smoothing window fill with sharp frames.

Two cooldowns keep the photo folder sane: no new photo within `cooldown_tiles`
crop widths of one already saved (the same crater seen from the next grid
point), and none within `cooldown_s` of the last one.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Config
from .controller import PanTilt

STEPS = [0.5, 1, 2, 5, 10]


# -- corners -------------------------------------------------------------------
Corner = tuple[float, float]  # (pan, tilt) in degrees


def check_corners(corners: list[Corner]) -> None:
    """Four corners in order around the board, not crossing over themselves."""
    if len(corners) != 4:
        raise ValueError(f"need 4 corners, got {len(corners)}")
    turns = []
    for i in range(4):
        (ax, ay), (bx, by), (cx, cy) = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
        turns.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        raise ValueError("corners cross over - mark them in order going around the board edge")
    area = abs(sum(corners[i][0] * corners[(i + 1) % 4][1] - corners[(i + 1) % 4][0] * corners[i][1]
                   for i in range(4))) / 2
    if area < 1.0:
        raise ValueError("corners are almost on top of each other - is that really the board?")


def save_corners(path: str | Path, corners: list[Corner]) -> None:
    check_corners(corners)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "taught": datetime.now().isoformat(timespec="seconds"),
        "corners": [[round(p, 2), round(t, 2)] for p, t in corners],
        "note": "(pan, tilt) with the crosshair on each board corner, in order around the edge",
    }, indent=2))


def load_corners(path: str | Path) -> list[Corner]:
    data = json.loads(Path(path).read_text())
    corners = [(float(p), float(t)) for p, t in data["corners"]]
    check_corners(corners)
    return corners


# -- grid ----------------------------------------------------------------------
@dataclass(frozen=True)
class Pose:
    index: int
    row: int
    col: int
    pan: float
    tilt: float


def tile_deg(hfov_deg: float, frame_w: int, frame_h: int, crop: float) -> float:
    """Angular size of the square the model classifies (same crop as preprocess())."""
    side_px = max(1, int(min(frame_w, frame_h) * crop))
    focal_px = (frame_w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return math.degrees(2 * math.atan(side_px / 2 / focal_px))


def _fractions(length_tiles: float, overlap: float) -> list[float]:
    """Positions 0..1 along an edge, inset half a tile from each end."""
    if length_tiles <= 1:
        return [0.5]
    n = math.ceil((length_tiles - 1) / (1 - overlap)) + 1
    inset = 0.5 / length_tiles
    return [inset + (1 - 2 * inset) * i / (n - 1) for i in range(n)]


def plan(corners: list[Corner], tile: float, overlap: float, serpentine: bool = True) -> list[Pose]:
    """Grid of poses covering the board, every crop square fully on the board.

    Interpolates bilinearly between the taught corners, so a board seen at an
    angle (a trapezoid in pan/tilt) is still covered edge to edge. Rows run
    along whichever pair of edges is closer to the pan direction.
    """
    check_corners(corners)
    c = list(corners)
    if abs(c[1][0] - c[0][0]) < abs(c[1][1] - c[0][1]):
        c = c[1:] + c[:1]  # make c0 -> c1 the more pan-like edge
    (p0, p1, p2, p3) = c

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    us = _fractions(max(dist(p0, p1), dist(p3, p2)) / tile, overlap)
    vs = _fractions(max(dist(p0, p3), dist(p1, p2)) / tile, overlap)

    poses = []
    for r, v in enumerate(vs):
        cols = list(enumerate(us))
        if serpentine and r % 2:
            cols.reverse()
        for col, u in cols:
            pan, tilt = (
                (1 - v) * ((1 - u) * p0[k] + u * p1[k]) + v * ((1 - u) * p3[k] + u * p2[k])
                for k in (0, 1)
            )
            poses.append(Pose(len(poses), r, col, pan, tilt))
    return poses


def outer_poses(poses: list[Pose]) -> list[Pose]:
    """The four extreme grid points, in order around the board (for --trace)."""
    last_row = max(p.row for p in poses)
    last_col = max(p.col for p in poses)
    want = [(0, 0), (0, last_col), (last_row, last_col), (last_row, 0)]
    by_rc = {(p.row, p.col): p for p in poses}
    out = []
    for rc in want:
        if by_rc[rc] not in out:
            out.append(by_rc[rc])
    return out


# -- cooldown ------------------------------------------------------------------
class Cooldown:
    """Decides whether a detection earns a photo."""

    def __init__(self, min_deg: float, min_s: float, max_photos: int = 0):
        self.min_deg = min_deg
        self.min_s = min_s
        self.max_photos = max_photos
        self.saved: list[Corner] = []
        self.last_t: float | None = None

    def check(self, pan: float, tilt: float, now: float) -> str:
        """'save', or why not: 'near' / 'wait' / 'cap'."""
        if self.max_photos and len(self.saved) >= self.max_photos:
            return "cap"
        if any(math.hypot(pan - p, tilt - t) < self.min_deg for p, t in self.saved):
            return "near"
        if self.last_t is not None and now - self.last_t < self.min_s:
            return "wait"
        return "save"

    def record(self, pan: float, tilt: float, now: float) -> None:
        self.saved.append((pan, tilt))
        self.last_t = now


# -- image helpers -------------------------------------------------------------
def _small_grey(frame):
    """16 x 16 block means: averages out sensor noise, keeps real motion."""
    import numpy as np

    h, w = (frame.shape[0] // 16) * 16, (frame.shape[1] // 16) * 16
    g = frame[:h, :w, :3].astype(np.float32).mean(axis=2)
    return g.reshape(h // 16, 16, w // 16, 16).mean(axis=(1, 3))


def frame_diff(a, b) -> float:
    """Mean grey-level change between two frames (0 = identical)."""
    import numpy as np

    return float(np.abs(_small_grey(a) - _small_grey(b)).mean())


def wait_still(grab, still_diff: float, timeout_s: float, clock=time.monotonic) -> bool:
    """Grab frames until two in a row match, or time runs out. True if still."""
    if still_diff <= 0 or timeout_s <= 0:
        return True
    end = clock() + timeout_s
    prev = grab()
    while clock() < end:
        cur = grab()
        if frame_diff(prev, cur) < still_diff:
            return True
        prev = cur
    return False


def center_crop(frame, crop: float):
    """The same square preprocess() hands the model, in colour."""
    h, w = frame.shape[:2]
    side = max(1, int(min(h, w) * crop))
    top, left = (h - side) // 2, (w - side) // 2
    return frame[top:top + side, left:left + side, :3]


def _save_photo(path: Path, rgb, quality: int = 92):
    from PIL import Image

    Image.fromarray(rgb).save(path, quality=quality)


# -- teach ---------------------------------------------------------------------
def cmd_teach(args, cfg: Config) -> int:
    """Aim the crosshair at each board corner with the arrow keys and mark it."""
    import curses

    from .stream import CameraStream

    corners_file = Path(args.file or cfg.sweep.corners_file)
    crop = args.crop if args.crop is not None else cfg.model.crop
    corners: list[Corner] = []
    state = {"step_i": STEPS.index(2), "msg": "", "saved": False}
    pt_ref: dict = {}

    def status():
        p, t = pt_ref["pt"].position if "pt" in pt_ref else (None, None)
        d = {"mode": "teach corners", "pan": p if p is not None else "-",
             "tilt": t if t is not None else "-", "corners": f"{len(corners)}/4", "crop": f"{crop:.2f}"}
        for i, (cp, ct) in enumerate(corners, 1):
            d[f"#{i}"] = f"pan {cp:.1f} tilt {ct:.1f}"
        return d

    stream = CameraStream(cfg.camera, status=status)
    url = stream.start()
    try:
        with PanTilt(cfg) as pt:
            pt_ref["pt"] = pt
            pt.center(smooth=False)

            def loop(scr):
                scr.keypad(True)
                scr.timeout(200)
                try:
                    curses.curs_set(0)
                except curses.error:
                    pass
                while True:
                    step = STEPS[state["step_i"]]
                    p, t = pt.position
                    nxt = (f"aim the crosshair at corner {len(corners) + 1} of 4 and press m"
                           if len(corners) < 4 else "all 4 marked - s saves, u undoes the last")
                    rows = [
                        "Teach the board corners",
                        f"view: {url}",
                        "",
                        "Put the green crosshair exactly on each corner of the printed map,",
                        "going around the edge in order (either direction, any start).",
                        "",
                        f"pan  = {p:6.1f}    tilt = {t:6.1f}    step = {step} deg",
                        "",
                        *[f"  corner {i}: pan {cp:6.1f}  tilt {ct:6.1f}"
                          for i, (cp, ct) in enumerate(corners, 1)],
                        "",
                        nxt,
                        "arrows: move   +/-: step   m/space: mark   u: undo   g: go to corners",
                        "s: save   q: quit",
                        "",
                        state["msg"],
                    ]
                    scr.erase()
                    for y, text in enumerate(rows):
                        try:
                            scr.addstr(y, 0, text)
                        except curses.error:
                            pass
                    scr.refresh()
                    k = scr.getch()
                    if k == -1:
                        continue
                    state["msg"] = ""
                    if k == curses.KEY_UP:
                        pt.nudge(dtilt=+step)
                    elif k == curses.KEY_DOWN:
                        pt.nudge(dtilt=-step)
                    elif k == curses.KEY_RIGHT:
                        pt.nudge(dpan=+step)
                    elif k == curses.KEY_LEFT:
                        pt.nudge(dpan=-step)
                    elif k in (ord("+"), ord("=")):
                        state["step_i"] = min(state["step_i"] + 1, len(STEPS) - 1)
                    elif k in (ord("-"), ord("_")):
                        state["step_i"] = max(state["step_i"] - 1, 0)
                    elif k in (ord("m"), ord("M"), ord(" ")):
                        if len(corners) < 4:
                            corners.append(pt.position)
                            state["msg"] = f"marked corner {len(corners)}"
                        else:
                            state["msg"] = "already have 4 - u to undo one"
                    elif k in (ord("u"), ord("U")):
                        if corners:
                            corners.pop()
                            state["msg"] = "removed the last corner"
                    elif k in (ord("g"), ord("G")):
                        for cp, ct in corners:  # re-check each mark in the preview
                            pt.move_to(pan=cp, tilt=ct, ease=True)
                            time.sleep(1.5)
                    elif k in (ord("s"), ord("S")):
                        try:
                            save_corners(corners_file, corners)
                        except ValueError as e:
                            state["msg"] = f"not saved: {e}"
                        else:
                            state["saved"] = True
                            return
                    elif k in (ord("q"), ord("Q"), 27):
                        return

            curses.wrapper(loop)
            pt.center()
    finally:
        stream.stop()

    if state["saved"]:
        print(f"saved {len(corners)} corners to {corners_file}")
        print("next: pantilt board-scan --crop 0.2 --dry-run")
        return 0
    print("quit without saving")
    return 1


# -- board-scan ----------------------------------------------------------------
def _plan_for(args, cfg: Config):
    s = cfg.sweep
    crop = args.crop if args.crop is not None else cfg.model.crop
    corners = load_corners(args.corners or s.corners_file)
    tile = tile_deg(s.hfov_deg, cfg.camera.preview_width, cfg.camera.preview_height, crop)
    poses = plan(corners, tile, s.overlap, serpentine=not args.one_way)
    return crop, corners, tile, poses


def _describe(cfg, crop, tile, poses, threshold, cooldown_deg, cooldown_s, speed) -> str:
    rows = max(p.row for p in poses) + 1
    cols = max(p.col for p in poses) + 1
    pans = [p.pan for p in poses]
    tilts = [p.tilt for p in poses]
    return "\n".join([
        f"crop         {crop:.2f} of the frame -> {tile:.1f} deg square on the board",
        f"grid         {cols} cols x {rows} rows = {len(poses)} poses  (overlap {cfg.sweep.overlap:.0%})",
        f"pan range    {min(pans):.1f} .. {max(pans):.1f} deg",
        f"tilt range   {min(tilts):.1f} .. {max(tilts):.1f} deg",
        f"trigger      {cfg.sweep.target} >= {threshold:.2f}",
        f"cooldown     {cooldown_deg:.1f} deg apart and {cooldown_s:.1f} s between photos",
        f"motion       {speed:.0f} deg/s, settle {cfg.sweep.settle_s:.1f}-{cfg.sweep.settle_max_s:.1f} s",
    ])


def cmd_board_scan(args, cfg: Config) -> int:
    from .classify import CLASS_NAMES

    s = cfg.sweep
    if s.target not in CLASS_NAMES:
        print(f"[sweep] target must be one of: {', '.join(CLASS_NAMES)}", file=sys.stderr)
        return 2
    try:
        crop, corners, tile, poses = _plan_for(args, cfg)
    except FileNotFoundError:
        print(f"no board corners at {args.corners or s.corners_file} - run `pantilt teach` first",
              file=sys.stderr)
        return 1
    except (ValueError, KeyError) as e:
        print(f"bad corners file: {e} - re-run `pantilt teach`", file=sys.stderr)
        return 1

    threshold = args.threshold if args.threshold is not None else s.threshold
    cooldown_deg = (args.cooldown_tiles if args.cooldown_tiles is not None else s.cooldown_tiles) * tile
    cooldown_s = args.cooldown_s if args.cooldown_s is not None else s.cooldown_s
    speed = args.speed or s.speed_dps

    print(_describe(cfg, crop, tile, poses, threshold, cooldown_deg, cooldown_s, speed))
    print()

    if args.dry_run:
        for p in poses[:10]:
            print(f"  {p.index:>3}  r{p.row} c{p.col}  ->  pan {p.pan:6.2f}  tilt {p.tilt:6.2f}")
        if len(poses) > 10:
            print(f"  ... {len(poses) - 10} more")
        return 0
    if args.trace:
        return _trace(cfg, crop, poses, speed)
    return _scan(args, cfg, crop, corners, tile, poses, threshold, cooldown_deg, cooldown_s, speed)


def _trace(cfg: Config, crop: float, poses: list[Pose], speed: float) -> int:
    """Visit the outermost grid points so you can check the red box stays on the board."""
    from .stream import CameraStream

    where = {"at": "-"}
    stream = CameraStream(cfg.camera, status=lambda: {"mode": "trace", "at": where["at"],
                                                      "crop": f"{crop:.2f}"})
    print(f"live view: {stream.start()}")
    print("Watch the red box at each stop: it should touch the board edge but stay on the board.")
    try:
        with PanTilt(cfg) as pt:
            pt.center(smooth=False)
            for p in outer_poses(poses):
                where["at"] = f"r{p.row} c{p.col}"
                pt.move_to(pan=p.pan, tilt=p.tilt, speed_dps=speed, ease=True)
                print(f"  r{p.row} c{p.col}  pan {p.pan:6.2f}  tilt {p.tilt:6.2f}")
                time.sleep(3)
            pt.center()
    finally:
        stream.stop()
    print("If the box hung off the board, re-teach with the crosshair a little further in, "
          "or lower --crop.")
    return 0


def _scan(args, cfg, crop, corners, tile, poses, threshold, cooldown_deg, cooldown_s,
          speed) -> int:
    from .classify import CLASS_NAMES, HailoClassifier, LiveClassifier
    from .cli import _Camera, _thumb_source

    s, m = cfg.sweep, cfg.model
    hef = Path(args.hef or m.hef_path)
    if not hef.exists():
        print(f"model not found: {hef}", file=sys.stderr)
        return 1
    try:
        import picamera2  # noqa: F401
        import hailo_platform  # noqa: F401
    except ImportError as e:
        print(f"{e.name} not found. Install with `sudo apt install python3-picamera2 python3-hailort` "
              "and create the venv with --system-site-packages.", file=sys.stderr)
        return 1

    target_i = CLASS_NAMES.index(s.target)
    run = Path(args.out or f"data/runs/board-{datetime.now():%Y%m%d-%H%M%S}")
    (run / "photos").mkdir(parents=True, exist_ok=True)
    cooldown = Cooldown(cooldown_deg, cooldown_s, args.max_photos if args.max_photos is not None
                        else s.max_photos)

    progress = {"pose": "-", "pass": 0, "state": "starting", "photos": 0}
    pt_ref: dict = {}
    cam_ref: dict = {}
    live = None

    def status():
        d = {"mode": "board scan", **progress}
        if live:
            d.update(live.status())
        d["crop"] = f"{crop:.2f}"
        if "pt" in pt_ref:
            p, t = pt_ref["pt"].position
            d["pan"] = p if p is not None else "-"
            d["tilt"] = t if t is not None else "-"
        return d

    (run / "run.json").write_text(json.dumps({
        "started": datetime.now().isoformat(timespec="seconds"),
        "corners": corners, "crop": crop, "tile_deg": tile, "poses": len(poses),
        "target": s.target, "threshold": threshold,
        "cooldown_deg": cooldown_deg, "cooldown_s": cooldown_s,
        "speed_dps": speed, "settle_s": s.settle_s, "settle_max_s": s.settle_max_s,
        "still_diff": s.still_diff, "hef": str(hef),
    }, indent=2))

    camera = None
    log = open(run / "poses.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["pass", "index", "row", "col", "pan", "tilt", "time", "settle_ms", "still",
                     f"p_{s.target.replace(' ', '_')}", "top_label", "top_p", "action", "photo",
                     *[f"p_{n.replace(' ', '_')}" for n in CLASS_NAMES]])
    t_start = time.time()
    try:
        with HailoClassifier(str(hef)) as clf, PanTilt(cfg) as pt:
            pt_ref["pt"] = pt
            camera = _Camera(cfg, preview=not args.no_preview, port=args.port, status=status,
                             thumb=_thumb_source(cam_ref, crop))
            cam_ref["camera"] = camera
            if camera.url:
                print(f"live view: {camera.url}")
            time.sleep(1)  # let auto-exposure settle
            live = LiveClassifier(camera.grab, clf.classify, smooth=m.smooth_frames, crop=crop,
                                  min_confidence=m.min_confidence)
            live.start()
            pt.center(smooth=False)
            try:
                n_pass = 0
                while args.passes == 0 or n_pass < args.passes:
                    n_pass += 1
                    progress["pass"] = n_pass
                    for pose in poses:
                        progress["pose"] = f"{pose.index + 1}/{len(poses)}"
                        _visit(pose, n_pass, pt, camera, live, cooldown, writer, log, run, progress,
                               s, crop, m.smooth_frames, target_i, threshold, speed)
            finally:
                live.stop()  # before the pipeline closes - see cmd_live
            pt.center()
    finally:
        if live:
            live.stop()
        if camera:
            camera.close()
        log.close()
        n = len(cooldown.saved)
        print(f"\n{n} photo{'' if n == 1 else 's'} in {time.time() - t_start:.0f}s -> {run}")
    if live and live.error:
        raise live.error
    return 0


def _visit(pose, n_pass, pt, camera, live, cooldown, writer, log, run, progress,
           s, crop, smooth, target_i, threshold, speed):
    from .classify import CLASS_NAMES

    progress["state"] = "moving"
    pt.move_to(pan=pose.pan, tilt=pose.tilt, speed_dps=speed, ease=True)

    progress["state"] = "settling"
    t0 = time.monotonic()
    time.sleep(s.settle_s)
    still = wait_still(camera.grab, s.still_diff, s.settle_max_s - s.settle_s)
    settle_ms = (time.monotonic() - t0) * 1000

    # Fill the smoothing window with frames taken after the mount stopped.
    progress["state"] = "classifying"
    start = live.latest.frames
    while live.latest.frames - start < smooth + 1:
        if live.error:
            raise live.error
        time.sleep(0.01)
    p = live.latest
    target_p = float(p.probs[target_i])

    action, name = "below", ""
    if target_p >= threshold:
        now = time.monotonic()
        action = cooldown.check(pose.pan, pose.tilt, now)
        if action == "save":
            frame = camera.grab()
            n = len(cooldown.saved) + 1
            stem = f"{n:03d}_p{pose.pan:05.1f}_t{pose.tilt:05.1f}_{s.target.replace(' ', '-')}{target_p:.2f}"
            name = f"{stem}.jpg"
            _save_photo(run / "photos" / name, frame[:, :, :3])
            _save_photo(run / "photos" / f"{stem}_crop.png", center_crop(frame, crop))
            cooldown.record(pose.pan, pose.tilt, now)
            progress["photos"] = len(cooldown.saved)
            action = "saved"

    writer.writerow([n_pass, pose.index, pose.row, pose.col, f"{pose.pan:.2f}", f"{pose.tilt:.2f}",
                     f"{time.time():.3f}", f"{settle_ms:.0f}", int(still), f"{target_p:.4f}",
                     p.label, f"{p.confidence:.4f}", action, name,
                     *[f"{v:.4f}" for v in p.probs]])
    log.flush()

    note = {"saved": f"  -> PHOTO {name}", "near": "  (cooldown: already photographed nearby)",
            "wait": "  (cooldown: too soon)", "cap": "  (photo limit reached)"}.get(action, "")
    shake = "" if still else "  [still shaking]"
    print(f"  [{progress['pose']}] pan {pose.pan:6.2f} tilt {pose.tilt:6.2f}  "
          f"{s.target} {target_p:5.2f}  top {CLASS_NAMES[p.index] if p.index >= 0 else '-':<14}"
          f"{shake}{note}")
