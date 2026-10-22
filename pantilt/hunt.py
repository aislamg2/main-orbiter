"""Find-then-confirm crater hunting: the default `pantilt board-scan`.

The grid scan judges each square once, wherever it happens to land, so a
crater sitting across a square's edge scores low and is missed. This does it
in two stages instead:

1. Find. Point at the board and slide the crop square across the still frame
   in small steps (a third of a square by default). Every crater ends up near
   the middle of some window. The camera doesn't move, so there's no blur, and
   on the Hailo a hundred windows take well under a second. Anything scoring
   at least `look_threshold` becomes a candidate.
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
    Cooldown, Corner, _save_photo, check_corners, load_step, plan, wait_still,
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
    action: str = ""
    photo: str = ""
    frame: object = field(default=None, repr=False)   # best confirm frame, until saved
    xy: tuple = (0, 0)                                # best square's centre in that frame


# -- the run -------------------------------------------------------------------
def plan_summary(cfg: Config, corners, step, crop: float) -> tuple[int, list, int, int]:
    w, h = cfg.camera.preview_width, cfg.camera.preview_height
    side = max(1, int(min(w, h) * crop))
    stride = max(1, round(side * cfg.sweep.find_stride))
    poses = find_poses(corners, step, w, h, side)
    counts = [len(windows(View(step, w, h, p.pan, p.tilt), corners, side, stride)) for p in poses]
    return side, poses, stride, sum(counts)


def describe(cfg: Config, crop: float, side: int, poses, stride: int, n_windows: int,
             threshold: float) -> str:
    s = cfg.sweep
    return "\n".join([
        "mode         find + confirm  (--grid for the old stop-and-go grid)",
        f"crop         {crop:.2f} of the frame -> {side} px square",
        f"find         {len(poses)} camera pose{'s' if len(poses) != 1 else ''}, "
        f"{n_windows} windows {stride} px apart",
        f"candidates   {s.target} >= {s.look_threshold:.2f} in the find stage, up to {s.max_candidates}",
        f"photo        {s.target} >= {threshold:.2f} after aiming straight at it "
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
    side, poses, stride, _ = plan_summary(cfg, corners, step, crop)
    square_deg = side / min(step["pan_px_per_deg"], step["tilt_px_per_deg"]) / (w / step["frame"][0])
    cooldown = Cooldown(cooldown_tiles * square_deg, cooldown_s,
                        args.max_photos if args.max_photos is not None else s.max_photos)

    run = Path(args.out or f"data/runs/board-{datetime.now():%Y%m%d-%H%M%S}")
    (run / "photos").mkdir(parents=True, exist_ok=True)
    (run / "run.json").write_text(json.dumps({
        "started": datetime.now().isoformat(timespec="seconds"), "mode": "find+confirm",
        "corners": corners, "crop": crop, "side_px": side, "stride_px": stride,
        "find_poses": [[p.pan, p.tilt] for p in poses], "target": s.target,
        "look_threshold": s.look_threshold, "threshold": threshold,
        "refine_steps": s.refine_steps, "find_frames": s.find_frames,
        "confirm_frames": s.confirm_frames, "max_candidates": s.max_candidates,
        "cooldown_deg": cooldown.min_deg, "cooldown_s": cooldown_s, "speed_dps": speed,
        "hef": str(hef),
    }, indent=2))

    progress = {"mode": "find + confirm", "stage": "starting", "photos": 0}
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
    cand_w.writerow(["pass", "id", "pan", "tilt", "find_p", "confirm_p", "re_aims", "action", "photo",
                     *names])

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

            n_pass = 0
            while args.passes == 0 or n_pass < args.passes:
                n_pass += 1
                progress["pass"] = n_pass

                # 1. find
                found, overviews = [], []
                for k, pose in enumerate(poses):
                    progress["stage"] = f"find {k + 1}/{len(poses)}"
                    still = aim(pose.pan, pose.tilt)
                    frames = [camera.grab() for _ in range(s.find_frames)]
                    view = View(step, frames[0].shape[1], frames[0].shape[0], *pt.position)
                    centres = windows(view, corners, side, stride)
                    t0 = time.time()
                    probs = score_windows(frames, centres, side, clf.classify)
                    for (x, y), p in zip(centres, probs):
                        pan, tilt = view.to_angle(x, y)
                        win_w.writerow([n_pass, k, x, y, f"{pan:.2f}", f"{tilt:.2f}",
                                        f"{p[target_i]:.4f}", CLASS_NAMES[int(np.argmax(p))],
                                        *[f"{v:.4f}" for v in p]])
                        if p[target_i] >= s.look_threshold:
                            found.append(Candidate(pan, tilt, float(p[target_i]), k, x, y))
                    win_log.flush()
                    best = max((float(p[target_i]) for p in probs), default=0.0)
                    print(f"find {k + 1}/{len(poses)}: {len(centres)} windows in "
                          f"{time.time() - t0:.1f}s, best {s.target} {best:.2f}"
                          f"{'' if still else '  [still shaking]'}")
                    overviews.append((frames[0], view))

                cands = pick_candidates(found, 0.5 * square_deg, s.max_candidates)
                print(f"{len(found)} windows over {s.look_threshold:.2f} -> {len(cands)} spots to check")

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
                    progress["stage"] = f"confirm {i}/{len(order)}"
                    _confirm(c, aim, camera, pt, step, corners, side, clf.classify, s, target_i)
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
                                     f"{c.confirm_p:.4f}", c.moves, action, c.photo,
                                     *[f"{v:.4f}" for v in c.probs]])
                    cand_log.flush()
                    note = {"saved": f"  -> PHOTO {c.photo}", "near-miss": "  (near miss, logged)",
                            "near": "  (already photographed nearby)", "wait": "  (cooldown)",
                            "cap": "  (photo limit reached)"}.get(action, "")
                    print(f"  [{i}/{len(order)}] pan {c.pan:6.2f} tilt {c.tilt:6.2f}  "
                          f"find {c.find_p:.2f} -> {c.confirm_p:.2f}{note}")
                    c.frame = None  # don't hold every frame in memory

                for k, (frame, view) in enumerate(overviews):
                    _overview(run / f"overview_p{n_pass}_{k}.jpg", frame, view, corners, side,
                              [c for c in order if c.find_pose == k])
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


def _confirm(c: Candidate, aim, camera, pt, step, corners, side, classify, s, target_i):
    """Aim at the candidate, find the best square around it, re-aim if it is far off."""
    pan, tilt = c.pan, c.tilt
    for attempt in range(s.refine_steps + 1):
        aim(pan, tilt)
        frames = [camera.grab() for _ in range(s.confirm_frames)]
        fh, fw = frames[0].shape[:2]
        view = View(step, fw, fh, *pt.position)
        cx, cy = int(view.cx), int(view.cy)
        (bx, by), probs = _refine(frames, view, corners, side, classify, target_i, cx, cy)
        c.confirm_p, c.frame, c.xy = float(probs[target_i]), frames[-1], (bx, by)
        c.probs = [float(v) for v in probs]
        c.pan, c.tilt = view.to_angle(bx, by)
        # Close to the middle already: the lens is sharpest there, good enough.
        if math.hypot(bx - cx, by - cy) <= side / 8 or attempt == s.refine_steps:
            break
        pan, tilt = c.pan, c.tilt
        c.moves += 1


def _overview(path: Path, frame, view: View, corners, side: int, cands: list[Candidate]):
    """The find frame with the board outline and every checked spot drawn on it.

    Green = photographed, orange = near miss, grey = skipped by a cooldown.
    """
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame[:, :, :3].copy())
    d = ImageDraw.Draw(img)
    pts = [view.to_px(p, t) for p, t in corners]
    d.line(pts + pts[:1], fill=(0, 200, 255), width=2)
    colours = {"saved": (0, 255, 0), "near-miss": (255, 170, 0)}
    h = side / 2
    for c in cands:
        col = colours.get(c.action, (160, 160, 160))
        x, y = view.to_px(c.pan, c.tilt)  # where confirm pinned it, not the first rough window
        d.rectangle([x - h, y - h, x + h, y + h], outline=col, width=3)
        label = f"{c.id}: {c.find_p:.2f} -> {c.confirm_p:.2f}"
        box = d.textbbox((x - h + 4, y - h + 4), label)
        d.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], fill=(0, 0, 0))
        d.text((x - h + 4, y - h + 4), label, fill=col)
    img.save(path, quality=90)
