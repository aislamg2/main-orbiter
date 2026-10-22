"""Scan-then-revisit crater hunting: the default `pantilt board-scan`.

Judging each grid square once, wherever it happens to land, misses a crater
sitting across a square's edge. This does it in two stages instead:

1. Scan. Walk the camera over the board grid (one crop square per stop, as in
   `--trace`). At each stop, once the mount is still, score the centre square
   and copies nudged a third of a square each way, so every crater is near the
   middle of some window, and each part of the board is judged near the lens
   centre where the picture is sharpest. Anything scoring at least
   `look_threshold` is a point of interest. (`--wide` does this from a single
   frame of the whole board instead, without walking the grid.)
2. Confirm. Aim the camera straight at each candidate, let it settle, and
   search for the best-scoring square around the middle (a quarter square
   each way, then an eighth, then a sixteenth - on the same frames, no moving).
   If the best square is well off-centre, re-aim there and check again, up to
   `refine_steps` times.
   A photo is saved only if the best score clears `threshold`; otherwise the
   candidate is logged as a near miss.

The board's corners (from `pantilt teach`) decide which windows count, and the
calibration (from `pantilt calibrate-step`) turns pixels into servo angles.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .board_scan import (
    Cooldown, Corner, _save_photo, check_corners, load_step, plan, tile_from_step, wait_still,
)
from .config import Config
from .controller import PanTilt


# -- geometry ------------------------------------------------------------------
class View:
    """Pixel <-> servo-angle mapping for one camera pose, from the calibration.

    pan_vec / tilt_vec say how far (in pixels) the scene slides per +1 degree.
    A board point d degrees away from where the camera points therefore shows
    up at centre - J.d. Linear, which is accurate to a few pixels across the
    frame - and the confirm stage re-aims at the target anyway.
    """

    def __init__(self, step: dict, frame_w: int, frame_h: int, pan: float, tilt: float):
        if "pan_vec" not in step or "tilt_vec" not in step:
            raise ValueError("calibration has no direction info - re-run `pantilt calibrate-step`")
        scale = frame_w / step["frame"][0]
        (a, c), (b, d) = step["pan_vec"], step["tilt_vec"]  # columns of J: (x, y) per degree
        self.j = (a * scale, b * scale, c * scale, d * scale)  # [[a b], [c d]]
        det = self.j[0] * self.j[3] - self.j[1] * self.j[2]
        if abs(det) < 1e-9:
            raise ValueError("pan and tilt move the image the same way - re-run calibrate-step")
        self.det = det
        self.cx, self.cy = frame_w / 2, frame_h / 2
        self.w, self.h = frame_w, frame_h
        self.pan, self.tilt = pan, tilt

    def to_px(self, pan: float, tilt: float) -> tuple[float, float]:
        a, b, c, d = self.j
        dp, dt = pan - self.pan, tilt - self.tilt
        return self.cx - (a * dp + b * dt), self.cy - (c * dp + d * dt)

    def to_angle(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d = self.j
        ex, ey = self.cx - x, self.cy - y
        return (self.pan + (d * ex - b * ey) / self.det,
                self.tilt + (-c * ex + a * ey) / self.det)


def inside(corners: list[Corner], pan: float, tilt: float) -> bool:
    """Point in the (convex) taught board outline."""
    signs = []
    for i in range(4):
        (ax, ay), (bx, by) = corners[i], corners[(i + 1) % 4]
        signs.append((bx - ax) * (tilt - ay) - (by - ay) * (pan - ax))
    return all(v >= 0 for v in signs) or all(v <= 0 for v in signs)


def window_on_board(view: View, corners: list[Corner], x: float, y: float, side: int) -> bool:
    h = side / 2
    if x - h < 0 or y - h < 0 or x + h > view.w or y + h > view.h:
        return False
    return all(inside(corners, *view.to_angle(x + dx, y + dy))
               for dx in (-h, h) for dy in (-h, h))


def windows(view: View, corners: list[Corner], side: int, stride: int) -> list[tuple[int, int]]:
    """Window centres on a lattice over the frame, keeping those fully on the board."""
    half = side // 2
    out = []
    for y in range(half, view.h - half + 1, stride):
        for x in range(half, view.w - half + 1, stride):
            if window_on_board(view, corners, x, y, side):
                out.append((x, y))
    return out


def find_poses(corners: list[Corner], step: dict, frame_w: int, frame_h: int, side: int):
    """Camera poses whose frames between them put a window over every board spot.

    Usually one: from the middle of the board the whole map fits in the frame.
    """
    scale = frame_w / step["frame"][0]
    ppd_p = step["pan_px_per_deg"] * scale
    ppd_t = step["tilt_px_per_deg"] * scale
    margin = side / 2  # keep a little slack inside the frame edge
    return plan(corners, ((frame_w - margin) / ppd_p, (frame_h - margin) / ppd_t), overlap=0.05)


def scan_poses(corners: list[Corner], step: dict, frame_w: int, frame_h: int, crop: float,
               overlap: float, max_gap: float):
    """The board grid the camera walks: one crop square per stop, as `--trace` shows."""
    return plan(corners, tile_from_step(step, frame_w, frame_h, crop), overlap,
                serpentine=True, max_gap=max_gap)


def local_windows(view: View, corners: list[Corner], side: int, stride: int) -> list[tuple[int, int]]:
    """The centre window plus copies nudged by `stride`, out to half a square each way.

    Neighbouring grid stops are at most a square apart, so between them these
    windows form one even lattice over the whole board.
    """
    n = int(side / 2 / stride + 1e-9)
    cx, cy = int(view.cx), int(view.cy)
    out = [(cx + i * stride, cy + j * stride) for j in range(-n, n + 1) for i in range(-n, n + 1)]
    return [xy for xy in out if window_on_board(view, corners, *xy, side)]


def crop_at(frame, x: int, y: int, side: int):
    h = side // 2
    return frame[y - h:y - h + side, x - h:x - h + side, :3]


def score_windows(frames, centres, side: int, classify) -> list:
    """Class probabilities per window, averaged over the frames given."""
    import numpy as np

    from .classify import preprocess

    out = []
    for x, y in centres:
        acc = None
        for f in frames:
            p = np.asarray(classify(preprocess(crop_at(f, x, y, side))), dtype=np.float32).reshape(-1)
            acc = p if acc is None else acc + p
        out.append(acc / len(frames))
    return out


@dataclass
class Watch:
    best: float           # highest smoothed target score seen
    probs: list           # per window: smoothed probabilities at that window's best moment
    frames: list          # the frames in the smoothing window at the overall best moment
    seen: int             # frames classified
    seconds: float
    rising: bool          # still climbing when time ran out


def watch(grab, score, target_i: int, smooth: int, min_s: float, max_s: float,
          rise_level: float, stop_at: float, on_update=None, clock=time.monotonic) -> Watch:
    """Keep classifying one view until the score settles, the way you would by hand.

    score(frame) gives one row of class probabilities per window. Scores are
    averaged over the last `smooth` frames, so one lucky frame can't pass. It
    always watches for min_s, then moves on once the score is low and flat. If
    the score is climbing - exposure still settling, a last bit of wobble dying
    away - or already worth a look (>= rise_level), it stays up to max_s and
    keeps the best steady score. It stops early once the score reaches stop_at.
    """
    import numpy as np

    from collections import deque

    buf: deque = deque(maxlen=max(1, smooth))
    history: list[float] = []
    best = -1.0
    best_rows = None
    best_frames: list = []
    t0 = last_rise = clock()
    seen = 0
    rising = False
    while True:
        frame = grab()
        buf.append((frame, np.asarray(score(frame), dtype=np.float32)))
        seen += 1
        avg = np.mean([p for _, p in buf], axis=0)            # windows x classes
        cur = float(avg[:, target_i].max()) if len(avg) else 0.0
        if best_rows is None:
            best_rows = avg.copy()
        else:  # each window keeps its own best moment
            better = avg[:, target_i] > best_rows[:, target_i]
            best_rows[better] = avg[better]
        if cur > best:
            best, best_frames = cur, [f for f, _ in buf]
        history.append(cur)
        now = clock()
        rising = len(history) > smooth and cur > history[-1 - smooth] + 0.02
        if rising:
            last_rise = now
        if on_update:
            on_update(avg[int(np.argmax(avg[:, target_i]))] if len(avg) else None, now - t0)
        if len(buf) < buf.maxlen:
            continue                      # fill the smoothing window first
        elapsed = now - t0
        if elapsed >= max_s or best >= stop_at:
            break
        if elapsed >= min_s:
            if cur < rise_level and not rising:
                break                     # nothing here
            if now - last_rise >= min_s:
                break                     # worth a look, but it has stopped improving
    return Watch(best=max(best, 0.0), probs=list(best_rows), frames=best_frames, seen=seen,
                 seconds=clock() - t0, rising=rising)


def pick_candidates(found: list, min_deg: float, limit: int) -> list:
    """Strongest first, dropping any within min_deg of a stronger one (non-max suppression)."""
    kept = []
    for c in sorted(found, key=lambda c: -c.find_p):
        if all(math.hypot(c.pan - k.pan, c.tilt - k.tilt) >= min_deg for k in kept):
            kept.append(c)
        if len(kept) >= limit:
            break
    return kept


@dataclass
class Candidate:
    pan: float
    tilt: float
    find_p: float
    find_pose: int
    x: int            # where it was in that find frame
    y: int
    id: int = -1
    confirm_p: float = 0.0
    probs: list = field(default_factory=list)
    moves: int = 0
    watched_s: float = 0.0
    action: str = ""
    photo: str = ""
    frame: object = field(default=None, repr=False)   # best confirm frame, until saved
    xy: tuple = (0, 0)                                # best square's centre in that frame


# -- exposure ------------------------------------------------------------------
def lock_exposure(picam2) -> dict:
    """Freeze exposure, gain and white balance at whatever auto mode chose just now.

    Every stop then sees the board at the same brightness, so no stop has to
    wait for the camera to re-adjust, and fading daylight can't drift the
    scores during a run. Call it while the camera looks at the board.
    """
    md = picam2.capture_metadata()
    settings = {
        "ExposureTime": int(md["ExposureTime"]),
        "AnalogueGain": float(md["AnalogueGain"]),
        "ColourGains": tuple(float(x) for x in md["ColourGains"]),
    }
    picam2.set_controls({"AeEnable": False, "AwbEnable": False, **settings})
    return settings


# -- the run -------------------------------------------------------------------
def plan_summary(cfg: Config, corners, step, crop: float,
                 wide: bool = False) -> tuple[int, list, int, int]:
    s = cfg.sweep
    w, h = cfg.camera.preview_width, cfg.camera.preview_height
    side = max(1, int(min(w, h) * crop))
    stride = max(1, round(side * s.find_stride))
    if wide:
        poses = find_poses(corners, step, w, h, side)
        pick = windows
    else:
        poses = scan_poses(corners, step, w, h, crop, s.overlap, s.max_gap)
        pick = local_windows
    counts = [len(pick(View(step, w, h, p.pan, p.tilt), corners, side, stride)) for p in poses]
    return side, poses, stride, sum(counts)


def describe(cfg: Config, crop: float, side: int, poses, stride: int, n_windows: int,
             threshold: float, wide: bool = False) -> str:
    s = cfg.sweep
    if wide:
        find = (f"find         {len(poses)} wide view{'s' if len(poses) != 1 else ''}, "
                f"{n_windows} windows {stride} px apart (--wide)")
    else:
        rows = max(p.row for p in poses) + 1
        cols = max(p.col for p in poses) + 1
        find = (f"scan         {cols} x {rows} = {len(poses)} stops over the board, "
                f"{n_windows} windows in all ({stride} px nudges)")
    mode = "wide view + revisit" if wide else "scan + revisit"
    return "\n".join([
        f"mode         {mode}  (--grid for the old one-look-per-square scan)",
        f"crop         {crop:.2f} of the frame -> {side} px square",
        find,
        f"interest     {s.target} >= {s.look_threshold:.2f} during the scan, up to {s.max_candidates} revisits",
        f"watch        {s.watch_min_s:.1f}-{s.watch_max_s:.1f} s per stop (longer while the score climbs), "
        f"up to {s.revisit_watch_max_s:.1f} s per revisit",
        f"photo        {s.target} >= {threshold:.2f} after going back and aiming straight at it "
        f"(up to {s.refine_steps} re-aims)",
    ])


def run_hunt(args, cfg: Config, corners: list[Corner], crop: float, threshold: float,
             cooldown_tiles: float, cooldown_s: float, speed: float) -> int:
    import numpy as np

    from .classify import CLASS_NAMES, HailoClassifier
    from .cli import _Camera, _thumb_source

    s, m = cfg.sweep, cfg.model
    check_corners(corners)
    step = load_step(s.step_file)
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
    w = cfg.camera.preview_width
    wide = bool(getattr(args, "wide", False))
    side, poses, stride, _ = plan_summary(cfg, corners, step, crop, wide)
    square_deg = side / min(step["pan_px_per_deg"], step["tilt_px_per_deg"]) / (w / step["frame"][0])
    cooldown = Cooldown(cooldown_tiles * square_deg, cooldown_s,
                        args.max_photos if args.max_photos is not None else s.max_photos)

    run = Path(args.out or f"data/runs/board-{datetime.now():%Y%m%d-%H%M%S}")
    (run / "photos").mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text(json.dumps({
        "started": datetime.now().isoformat(timespec="seconds"), "mode": "wide+confirm" if wide else "scan+revisit",
        "corners": corners, "crop": crop, "side_px": side, "stride_px": stride,
        "find_poses": [[p.pan, p.tilt] for p in poses], "target": s.target,
        "look_threshold": s.look_threshold, "threshold": threshold,
        "refine_steps": s.refine_steps, "find_frames": s.find_frames,
        "confirm_frames": s.confirm_frames, "max_candidates": s.max_candidates,
        "cooldown_deg": cooldown.min_deg, "cooldown_s": cooldown_s, "speed_dps": speed,
        "hef": str(hef),
    }, indent=2))

    progress = {"mode": "scan + revisit", "stage": "starting", "photos": 0}

    def show(probs, where: str) -> None:
        """Put the latest scores on the browser live view."""
        order = np.argsort(probs)[::-1]
        progress[f"{s.target} %"] = float(probs[target_i]) * 100
        progress["top"] = f"{CLASS_NAMES[order[0]]} {probs[order[0]] * 100:.0f}%"
        progress["#2"] = f"{CLASS_NAMES[order[1]]} {probs[order[1]] * 100:.0f}%"
        progress["scored"] = where
    pt_ref: dict = {}
    cam_ref: dict = {}

    def status():
        d = dict(progress)
        d["crop"] = f"{crop:.2f}"
        if "pt" in pt_ref:
            p, t = pt_ref["pt"].position
            d["pan"] = p if p is not None else "-"
            d["tilt"] = t if t is not None else "-"
        return d

    names = [f"p_{n.replace(' ', '_')}" for n in CLASS_NAMES]
    win_log = open(run / "windows.csv", "w", newline="")
    win_w = csv.writer(win_log)
    win_w.writerow(["pass", "find_pose", "x", "y", "pan", "tilt", f"p_{s.target.replace(' ', '_')}",
                    "top_label", *names])
    cand_log = open(run / "candidates.csv", "w", newline="")
    cand_w = csv.writer(cand_log)
    cand_w.writerow(["pass", "id", "pan", "tilt", "find_p", "confirm_p", "re_aims", "watched_s",
                     "action", "photo", *names])

    camera = None
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
            pt.center(smooth=False)

            def aim(pan, tilt):
                pt.move_to(pan=pan, tilt=tilt, speed_dps=speed, ease=True)
                time.sleep(s.settle_s)
                return wait_still(camera.grab, s.still_diff, s.settle_max_s - s.settle_s)

            mid = (sum(p for p, _ in corners) / 4, sum(t for _, t in corners) / 4)
            if getattr(args, "lock_exposure", False):
                progress["stage"] = "locking exposure"
                aim(*mid)
                time.sleep(1.5)  # let auto-exposure settle on the board itself
                locked = lock_exposure(camera.cam)
                time.sleep(0.5)  # a few frames for the new controls to take effect
                info = json.loads((run / "run.json").read_text())
                info["exposure_locked"] = locked
                (run / "run.json").write_text(json.dumps(info, indent=2))
                print(f"exposure locked: {locked['ExposureTime'] / 1000:.1f} ms, "
                      f"gain {locked['AnalogueGain']:.2f}")
                if locked["AnalogueGain"] > 3.0:
                    print("warning: gain is high - the board is underlit, scores will suffer. "
                          "Add light.", file=sys.stderr)

            n_pass = 0
            while args.passes == 0 or n_pass < args.passes:
                n_pass += 1
                progress["pass"] = n_pass

                # 0. one picture of the whole board, to draw the results on
                found, overviews = [], []
                if not wide:
                    progress["stage"] = "overview"
                    aim(*mid)
                    frame = camera.grab()
                    overviews.append((frame, View(step, frame.shape[1], frame.shape[0], *pt.position)))

                # 1. scan (or, with --wide, look at the whole board at once)
                label = "find" if wide else "scan"
                for k, pose in enumerate(poses):
                    progress["stage"] = f"{label} {k + 1}/{len(poses)}"
                    still = aim(pose.pan, pose.tilt)
                    first = camera.grab()
                    view = View(step, first.shape[1], first.shape[0], *pt.position)
                    centres = (windows if wide else local_windows)(view, corners, side, stride)
                    t0 = time.time()
                    where = f"{label} {k + 1}/{len(poses)}"
                    seen = None
                    if centres:
                        seen = watch(
                            camera.grab, lambda f: score_windows([f], centres, side, clf.classify),
                            target_i, s.find_frames, s.watch_min_s, s.watch_max_s,
                            rise_level=s.look_threshold / 2, stop_at=threshold,
                            on_update=lambda p, t: p is not None and show(p, f"{where} watching {t:.1f}s"))
                    probs = seen.probs if seen else []
                    frames = seen.frames if seen and seen.frames else [first]
                    for (x, y), p in zip(centres, probs):
                        pan, tilt = view.to_angle(x, y)
                        win_w.writerow([n_pass, k, x, y, f"{pan:.2f}", f"{tilt:.2f}",
                                        f"{p[target_i]:.4f}", CLASS_NAMES[int(np.argmax(p))],
                                        *[f"{v:.4f}" for v in p]])
                        if p[target_i] >= s.look_threshold:
                            found.append(Candidate(pan, tilt, float(p[target_i]), k, x, y))
                    win_log.flush()
                    best_i = int(np.argmax([p[target_i] for p in probs])) if probs else -1
                    best = float(probs[best_i][target_i]) if probs else 0.0
                    if probs:
                        show(probs[best_i], f"{label} {k + 1}/{len(poses)} (best window)")
                    progress["interest"] = len(pick_candidates(found, 0.5 * square_deg, s.max_candidates))
                    shake = "" if still else "  [still shaking]"
                    if seen and seen.rising:
                        shake += "  [still rising when time ran out]"
                    if wide:
                        print(f"find {k + 1}/{len(poses)}: {len(centres)} windows, watched "
                              f"{time.time() - t0:.1f}s, best {s.target} {best:.2f}{shake}")
                        overviews.append((frames[0], view))
                    else:
                        top = CLASS_NAMES[int(np.argmax(probs[best_i]))] if probs else "-"
                        flag = "  <- point of interest" if best >= s.look_threshold else ""
                        print(f"  scan [{k + 1}/{len(poses)}] pan {pose.pan:6.2f} tilt {pose.tilt:6.2f}  "
                              f"best {s.target} {best:.2f}  (top {top}, {time.time() - t0:.1f}s)"
                              f"{shake}{flag}")

                cands = pick_candidates(found, 0.5 * square_deg, s.max_candidates)
                print(f"\n{len(found)} windows over {s.look_threshold:.2f} -> "
                      f"{len(cands)} point{'' if len(cands) == 1 else 's'} of interest to go back to")

                # 2. confirm, nearest first so the mount doesn't zig-zag
                here = pt.position
                order = []
                todo = list(cands)
                while todo:
                    nxt = min(todo, key=lambda c: math.hypot(c.pan - here[0], c.tilt - here[1]))
                    todo.remove(nxt)
                    order.append(nxt)
                    here = (nxt.pan, nxt.tilt)

                for i, c in enumerate(order, 1):
                    c.id = i
                    progress["stage"] = f"revisit {i}/{len(order)}"
                    where = f"revisit {i}/{len(order)}"
                    _confirm(c, aim, camera, pt, step, corners, side, clf.classify, s, target_i,
                             threshold, lambda p, t, w=where: show(p, f"{w} watching {t:.1f}s"))
                    show(np.asarray(c.probs), where)
                    action = "near-miss"
                    if c.confirm_p >= threshold:
                        now = time.monotonic()
                        action = cooldown.check(c.pan, c.tilt, now)
                        if action == "save":
                            frame = c.frame
                            n = len(cooldown.saved) + 1
                            stem = (f"{n:03d}_p{c.pan:05.1f}_t{c.tilt:05.1f}_"
                                    f"{s.target.replace(' ', '-')}{c.confirm_p:.2f}")
                            c.photo = f"{stem}.jpg"
                            _save_photo(run / "photos" / c.photo, frame[:, :, :3])
                            _save_photo(run / "photos" / f"{stem}_crop.png",
                                        crop_at(frame, *c.xy, side))
                            cooldown.record(c.pan, c.tilt, now)
                            progress["photos"] = len(cooldown.saved)
                            action = "saved"
                    c.action = action
                    cand_w.writerow([n_pass, c.id, f"{c.pan:.2f}", f"{c.tilt:.2f}", f"{c.find_p:.4f}",
                                     f"{c.confirm_p:.4f}", c.moves, f"{c.watched_s:.1f}", action, c.photo,
                                     *[f"{v:.4f}" for v in c.probs]])
                    cand_log.flush()
                    note = {"saved": f"  -> PHOTO {c.photo}", "near-miss": "  (near miss, logged)",
                            "near": "  (already photographed nearby)", "wait": "  (cooldown)",
                            "cap": "  (photo limit reached)"}.get(action, "")
                    print(f"  [{i}/{len(order)}] pan {c.pan:6.2f} tilt {c.tilt:6.2f}  "
                          f"find {c.find_p:.2f} -> {c.confirm_p:.2f}  ({c.watched_s:.1f}s){note}")
                    c.frame = None  # don't hold every frame in memory

                for k, (frame, view) in enumerate(overviews):
                    mine = order if not wide else [c for c in order if c.find_pose == k]
                    _overview(run / f"overview_p{n_pass}_{k}.jpg", frame, view, corners, side,
                              mine, [] if wide else poses)
            pt.center()
    finally:
        if camera:
            camera.close()
        win_log.close()
        cand_log.close()
        n = len(cooldown.saved)
        print(f"\n{n} photo{'' if n == 1 else 's'} in {time.time() - t_start:.0f}s -> {run}")
    return 0


def _refine(frames, view, corners, side, classify, target_i, x0, y0):
    """Coarse-to-fine search for the best-scoring square around (x0, y0), no moving.

    Tries the 8 neighbours a quarter square away, moves to the best, and halves
    the step whenever the centre wins, down to a sixteenth of a square.
    """
    scores: dict = {}

    def score(x, y):
        if (x, y) not in scores:
            if not window_on_board(view, corners, x, y, side):
                scores[(x, y)] = None
            else:
                scores[(x, y)] = score_windows(frames, [(x, y)], side, classify)[0]
        return scores[(x, y)]

    best = (x0, y0)
    if score(*best) is None:
        return best, score_windows(frames, [best], side, classify)[0]
    step = max(1, side // 4)
    while step >= max(1, side // 16):
        around = [(best[0] + dx, best[1] + dy) for dy in (-step, 0, step) for dx in (-step, 0, step)]
        ok = [xy for xy in around if score(*xy) is not None]
        top = max(ok, key=lambda xy: scores[xy][target_i])
        if top == best:
            step //= 2
        else:
            best = top
    return best, scores[best]


def _confirm(c: Candidate, aim, camera, pt, step, corners, side, classify, s, target_i,
             threshold: float = 1.0, on_update=None):
    """Aim at the candidate, watch it settle, find the best square around it, re-aim if far off."""
    pan, tilt = c.pan, c.tilt
    for attempt in range(s.refine_steps + 1):
        aim(pan, tilt)
        first = camera.grab()
        fh, fw = first.shape[:2]
        view = View(step, fw, fh, *pt.position)
        cx, cy = int(view.cx), int(view.cy)
        # A revisit always deserves a proper look: rise_level 1 keeps it watching
        # until the score stops improving (or reaches the photo threshold).
        seen = watch(camera.grab, lambda f: score_windows([f], [(cx, cy)], side, classify),
                     target_i, s.confirm_frames, s.watch_min_s, s.revisit_watch_max_s,
                     rise_level=1.0, stop_at=threshold, on_update=on_update)
        frames = seen.frames or [first]
        c.watched_s += seen.seconds
        (bx, by), probs = _refine(frames, view, corners, side, classify, target_i, cx, cy)
        c.confirm_p, c.frame, c.xy = float(probs[target_i]), frames[-1], (bx, by)
        c.probs = [float(v) for v in probs]
        c.pan, c.tilt = view.to_angle(bx, by)
        # Close to the middle already: the lens is sharpest there, good enough.
        if math.hypot(bx - cx, by - cy) <= side / 8 or attempt == s.refine_steps:
            break
        pan, tilt = c.pan, c.tilt
        c.moves += 1


def _overview(path: Path, frame, view: View, corners, side: int, cands: list[Candidate],
              scan: list | None = None):
    """The find frame with the board outline and every checked spot drawn on it.

    Blue = board outline, thin magenta = the scan stops, green = photographed, orange = near miss,
    grey = skipped by a cooldown.
    """
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame[:, :, :3].copy())
    d = ImageDraw.Draw(img)
    pts = [view.to_px(p, t) for p, t in corners]
    d.line(pts + pts[:1], fill=(0, 200, 255), width=2)
    colours = {"saved": (0, 255, 0), "near-miss": (255, 170, 0)}
    h = side / 2
    for p in scan or []:
        x, y = view.to_px(p.pan, p.tilt)
        d.rectangle([x - h, y - h, x + h, y + h], outline=(255, 0, 255), width=1)
    for c in cands:
        col = colours.get(c.action, (160, 160, 160))
        x, y = view.to_px(c.pan, c.tilt)  # where confirm pinned it, not the first rough window
        d.rectangle([x - h, y - h, x + h, y + h], outline=col, width=3)
        label = f"{c.id}: {c.find_p:.2f} -> {c.confirm_p:.2f}"
        box = d.textbbox((x - h + 4, y - h + 4), label)
        d.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], fill=(0, 0, 0))
        d.text((x - h + 4, y - h + 4), label, fill=col)
    img.save(path, quality=90)
