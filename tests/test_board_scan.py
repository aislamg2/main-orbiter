"""Hardware-free tests for teach / board-scan: geometry, cooldown, stillness."""

import math

import numpy as np
import pytest

import pantilt.controller as controller
from pantilt import Config, PanTilt, PCA9685, load_config
from pantilt.board_scan import (
    Cooldown, center_crop, check_corners, frame_diff, image_shift, load_corners, load_step,
    outer_poses, plan, save_corners, save_step, tile_deg, tile_from_step, wait_still,
)
from pantilt.cli import build_parser

SQUARE = [(60, 110), (120, 110), (120, 70), (60, 70)]  # 60 x 40 deg, in order around


def inside(corners, pan, tilt):
    """Point-in-convex-quad test."""
    signs = []
    for i in range(4):
        (ax, ay), (bx, by) = corners[i], corners[(i + 1) % 4]
        signs.append((bx - ax) * (tilt - ay) - (by - ay) * (pan - ax) >= 0)
    return all(signs) or not any(signs)


def test_tile_deg_matches_lens():
    # Full-height crop on a 67 deg lens at 16:9 is the ~41 deg vertical FOV.
    assert tile_deg(67, 1280, 720, 1.0) == pytest.approx(41.0, abs=0.5)
    assert tile_deg(67, 1280, 720, 0.2) == pytest.approx(8.5, abs=0.1)


def test_plan_stays_inset_by_half_a_tile():
    tile = 10
    poses = plan(SQUARE, tile, overlap=0.25)
    pans = [p.pan for p in poses]
    tilts = [p.tilt for p in poses]
    assert min(pans) == pytest.approx(65) and max(pans) == pytest.approx(115)
    assert min(tilts) == pytest.approx(75) and max(tilts) == pytest.approx(105)
    # Neighbouring poses never further apart than (1 - overlap) tiles.
    xs = sorted(set(round(p, 6) for p in pans))
    assert max(b - a for a, b in zip(xs, xs[1:])) <= tile * 0.75 + 1e-9


def test_plan_on_a_trapezoid_stays_on_board():
    trap = [(70, 108), (110, 108), (120, 72), (60, 72)]  # far edge looks narrower
    for p in plan(trap, 6, 0.2):
        assert inside(trap, p.pan, p.tilt)


def test_plan_is_serpentine_and_rows_follow_pan():
    # Start on a vertical edge: rows should still run along pan.
    rotated = SQUARE[1:] + SQUARE[:1]
    poses = plan(rotated, 10, 0.25)
    row0 = [p for p in poses if p.row == 0]
    row1 = [p for p in poses if p.row == 1]
    assert len({round(p.tilt, 6) for p in row0}) == 1  # a row is one tilt
    assert row0[0].col == 0 and row1[0].col == max(p.col for p in row1)  # snakes back
    assert [p.index for p in poses] == list(range(len(poses)))
    one_way = plan(rotated, 10, 0.25, serpentine=False)
    assert [p.col for p in one_way if p.row == 1][0] == 0


def test_board_smaller_than_a_tile_gives_one_pose():
    poses = plan([(88, 92), (92, 92), (92, 88), (88, 88)], tile=10, overlap=0.25)
    assert len(poses) == 1
    assert (poses[0].pan, poses[0].tilt) == pytest.approx((90, 90))


def test_outer_poses_are_the_grid_corners():
    poses = plan(SQUARE, 10, 0.25)
    outer = outer_poses(poses)
    assert len(outer) == 4
    assert {(round(p.pan), round(p.tilt)) for p in outer} == {(65, 105), (115, 105), (115, 75), (65, 75)}


def test_crossed_or_missing_corners_rejected():
    with pytest.raises(ValueError, match="cross"):
        check_corners([SQUARE[0], SQUARE[2], SQUARE[1], SQUARE[3]])
    with pytest.raises(ValueError, match="4 corners"):
        check_corners(SQUARE[:3])
    with pytest.raises(ValueError, match="top of each other"):
        check_corners([(90, 90), (90.5, 90), (90.5, 90.5), (90, 90.5)])


def test_corners_round_trip(tmp_path):
    f = tmp_path / "sub" / "corners.json"
    save_corners(f, SQUARE)
    assert load_corners(f) == [(float(p), float(t)) for p, t in SQUARE]


def test_cooldown_distance_time_and_cap():
    cd = Cooldown(min_deg=8, min_s=3, max_photos=2)
    assert cd.check(90, 90, now=0) == "save"
    cd.record(90, 90, now=0)
    assert cd.check(95, 90, now=10) == "near"     # same crater, next grid point
    assert cd.check(110, 90, now=1) == "wait"     # far enough, too soon
    assert cd.check(110, 90, now=5) == "save"
    cd.record(110, 90, now=5)
    assert cd.check(130, 90, now=50) == "cap"


def test_frame_diff_ignores_noise_but_sees_motion():
    rng = np.random.default_rng(0)
    scene = np.kron(rng.integers(0, 256, (45, 80)), np.ones((16, 16)))[..., None].repeat(3, 2)
    noisy = lambda img: np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    a, b = noisy(scene), noisy(scene)
    shifted = noisy(np.roll(scene, 24, axis=1))
    assert frame_diff(a, b) < 1.0
    assert frame_diff(a, shifted) > 10


def test_wait_still_until_frames_match():
    still = np.zeros((32, 32, 3), np.uint8)
    frames = iter([np.full_like(still, v) for v in (0, 80, 160)] + [still] * 5)
    t = iter(range(100))
    assert wait_still(lambda: next(frames), 2.0, 50, clock=lambda: next(t))
    shaking = iter(np.full((32, 32, 3), v % 2 * 200, np.uint8) for v in range(100))
    t = iter(range(100))
    assert not wait_still(lambda: next(shaking), 2.0, 5, clock=lambda: next(t))
    assert wait_still(None, 0, 5)  # check disabled


def test_center_crop_matches_preprocess_square():
    frame = np.zeros((720, 1280, 4), np.uint8)
    frame[288:432, 568:712] = 255
    tile = center_crop(frame, 0.2)
    assert tile.shape == (144, 144, 3) and tile.min() == 255


def test_ease_move_lands_on_target(monkeypatch):
    monkeypatch.setattr(controller.time, "sleep", lambda s: None)

    class Bus:
        regs = {0x00: 0x11}
        def write_byte_data(self, a, r, v): self.regs[r] = v
        def read_byte_data(self, a, r): return self.regs.get(r, 0)
        def write_i2c_block_data(self, a, r, d): pass
        def close(self): pass

    pt = PanTilt(Config(), driver=PCA9685(smbus=Bus()))
    pt.jump(pan=60, tilt=60)
    seen = []
    orig = pt.pan.set
    pt.pan.set = lambda a: seen.append(a) or orig(a)
    assert pt.move_to(pan=120, tilt=100, ease=True) == (120, 100)
    assert seen == sorted(seen)                        # never overshoots or backs up
    assert seen[1] - seen[0] < seen[len(seen) // 2] - seen[len(seen) // 2 - 1]  # slow start


def test_sweep_config_and_parser(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[sweep]\nthreshold = 0.8\ncooldown_s = 1\n')
    cfg = load_config(f)
    assert cfg.sweep.threshold == 0.8 and cfg.sweep.cooldown_s == 1
    f.write_text('[sweep]\noverlap = 1.5\n')
    with pytest.raises(ValueError):
        load_config(f)
    args = build_parser().parse_args(["board-scan", "--crop", "0.2", "--threshold", "0.7", "--passes", "0"])
    assert (args.crop, args.threshold, args.passes) == (0.2, 0.7, 0)
    assert build_parser().parse_args(["teach"]).func.__name__ == "cmd_teach"


def test_full_scan_with_fake_hardware(tmp_path, monkeypatch):
    """Runs the real board-scan loop end to end: camera, Hailo and servos faked."""
    import sys
    import types

    import pantilt.board_scan as bs
    import pantilt.classify as classify
    import pantilt.cli as cli
    from pantilt.classify import CLASS_NAMES

    for name in ("picamera2", "hailo_platform"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(bs.time, "sleep", lambda s: None)
    monkeypatch.setattr(controller.time, "sleep", lambda s: None)

    class Bus:
        def __init__(self): self.regs = {0x00: 0x11}
        def write_byte_data(self, a, r, v): self.regs[r] = v
        def read_byte_data(self, a, r): return self.regs.get(r, 0)
        def write_i2c_block_data(self, a, r, d): pass
        def close(self): pass

    real_pantilt = bs.PanTilt
    monkeypatch.setattr(bs, "PanTilt", lambda cfg: real_pantilt(cfg, driver=PCA9685(smbus=Bus())))

    frame = np.full((720, 1280, 4), 128, np.uint8)

    class FakeCamera:
        url = None
        def __init__(self, *a, **k): pass
        def grab(self): return frame
        def close(self): pass

    monkeypatch.setattr(cli, "_Camera", FakeCamera)

    crater = CLASS_NAMES.index("crater")
    calls = {"n": 0}

    class FakeHailo:
        def __init__(self, hef): pass
        def __enter__(self): return self
        def __exit__(self, *e): pass
        def classify(self, x):
            calls["n"] += 1
            p = np.full(len(CLASS_NAMES), 0.01, np.float32)
            p[crater] = 0.9  # every pose looks like a crater
            return p

    monkeypatch.setattr(classify, "HailoClassifier", FakeHailo)

    corners = tmp_path / "corners.json"
    save_corners(corners, SQUARE)
    hef = tmp_path / "m.hef"
    hef.write_bytes(b"")
    out = tmp_path / "run"
    cfg = Config()
    args = build_parser().parse_args([
        "board-scan", "--crop", "0.2", "--corners", str(corners), "--hef", str(hef),
        "--out", str(out), "--cooldown-tiles", "1.2", "--no-preview"])
    assert bs.cmd_board_scan(args, cfg) == 0

    import csv
    import json

    rows = list(csv.DictReader((out / "poses.csv").open()))
    n_poses = json.loads((out / "run.json").read_text())["poses"]
    assert len(rows) == n_poses
    saved = [r for r in rows if r["action"] == "saved"]
    assert saved and len(saved) < len(rows)          # cooldown_tiles stopped duplicates
    assert {r["action"] for r in rows} <= {"saved", "near"}
    photos = sorted(p.name for p in (out / "photos").iterdir())
    assert len(photos) == 2 * len(saved)             # full frame + crop for each


def _spacings(values):
    xs = sorted(set(round(v, 6) for v in values))
    return [b - a for a, b in zip(xs, xs[1:])]


def test_zero_overlap_exact_fit_butts_squares_together():
    # 60 x 40 deg board, 10 deg squares: exactly 6 x 4, centres 10 apart.
    poses = plan(SQUARE, 10, overlap=0)
    assert len(poses) == 24
    assert all(d == pytest.approx(10) for d in _spacings(p.pan for p in poses))
    assert all(d == pytest.approx(10) for d in _spacings(p.tilt for p in poses))


def test_zero_overlap_leftover_is_spread_never_a_gap():
    # 63 deg across with 10 deg squares: 7 squares (6 would leave a gap), evenly spaced.
    board = [(60, 110), (123, 110), (123, 70), (60, 70)]
    poses = plan(board, 10, overlap=0)
    cols = {round(p.pan, 6) for p in poses}
    assert len(cols) == 7
    gaps = _spacings(p.pan for p in poses)
    assert max(gaps) <= 10 + 1e-9 and max(gaps) - min(gaps) < 1e-5
    assert min(cols) == pytest.approx(65) and max(cols) == pytest.approx(118)


def test_per_axis_tile_sizes():
    poses = plan(SQUARE, (10, 5), overlap=0)  # tilt squares half as tall in servo degrees
    assert len({round(p.pan, 6) for p in poses}) == 6
    assert len({round(p.tilt, 6) for p in poses}) == 8
    assert min(p.tilt for p in poses) == pytest.approx(72.5)


def test_image_shift_measures_known_move():
    rng = np.random.default_rng(1)
    big = np.kron(rng.integers(0, 256, (120, 200)), np.ones((8, 8)))
    a = big[100:820, 200:1480][..., None].repeat(3, 2).astype(np.uint8)
    b = big[100 + 12:820 + 12, 200 + 70:1480 + 70][..., None].repeat(3, 2).astype(np.uint8)
    dy, dx, q = image_shift(a, b)
    assert math.hypot(dy, dx) == pytest.approx(math.hypot(12, 70), abs=1.5)
    assert q > 0.05
    flat = np.full_like(a, 100)
    assert image_shift(flat, flat)[2] < image_shift(a, b)[2]


def test_step_file_round_trip_and_scaling(tmp_path):
    f = tmp_path / "step.json"
    save_step(f, 1280, 720, pan_px_per_deg=18.0, tilt_px_per_deg=16.0)
    step = load_step(f)
    tp, tt = tile_from_step(step, 1280, 720, 0.2)   # 144 px square
    assert (tp, tt) == pytest.approx((8.0, 9.0))
    # Preview doubled in size: same angles, the square has twice the pixels.
    assert tile_from_step(step, 2560, 1440, 0.2) == pytest.approx((8.0, 9.0))


def test_trace_and_calibrate_parse():
    p = build_parser()
    a = p.parse_args(["board-scan", "--trace"])
    assert a.trace and not a.trace_edges and a.hold is None
    assert p.parse_args(["board-scan", "--trace-edges", "--hold", "2"]).hold == 2
    c = p.parse_args(["calibrate-step", "--crop", "0.2"])
    assert c.func.__name__ == "cmd_calibrate_step" and c.delta == 5.0 and c.reps == 3


def _fake_rig(monkeypatch, pan_ppd, tilt_ppd):
    """Servos + a camera whose image slides over a texture as the servos move."""
    import pantilt.board_scan as bs
    import pantilt.stream as stream

    monkeypatch.setattr(bs.time, "sleep", lambda s: None)
    monkeypatch.setattr(controller.time, "sleep", lambda s: None)

    class Bus:
        def __init__(self): self.regs = {0x00: 0x11}
        def write_byte_data(self, a, r, v): self.regs[r] = v
        def read_byte_data(self, a, r): return self.regs.get(r, 0)
        def write_i2c_block_data(self, a, r, d): pass
        def close(self): pass

    rig = {}
    real_pantilt = bs.PanTilt

    def make_pt(cfg):
        rig["pt"] = real_pantilt(cfg, driver=PCA9685(smbus=Bus()))
        return rig["pt"]

    monkeypatch.setattr(bs, "PanTilt", make_pt)
    texture = np.kron(np.random.default_rng(2).integers(0, 256, (400, 600)), np.ones((6, 6)))

    class Cam:
        def capture_array(self, name):
            pan, tilt = rig["pt"].position if "pt" in rig else (90, 90)
            x = int(round(800 + (pan - 90) * pan_ppd))
            y = int(round(700 - (tilt - 90) * tilt_ppd))
            return texture[y:y + 720, x:x + 1280][..., None].repeat(3, 2).astype(np.uint8)

    class FakeStream:
        def __init__(self, cam, status=None, **k):
            self.picam2 = Cam()
            self.status = status
        def start(self):
            assert isinstance(self.status(), dict)  # what the browser HUD polls
            return "http://pi:8000/"
        def stop(self): pass

    monkeypatch.setattr(stream, "CameraStream", FakeStream)
    return bs, rig


def test_calibrate_step_recovers_true_scale(tmp_path, monkeypatch):
    bs, _ = _fake_rig(monkeypatch, pan_ppd=17.0, tilt_ppd=21.0)
    cfg = Config()
    cfg.sweep.corners_file = str(tmp_path / "none.json")
    step_file = tmp_path / "step.json"
    args = build_parser().parse_args(["calibrate-step", "--crop", "0.2", "--file", str(step_file)])
    assert bs.cmd_calibrate_step(args, cfg) == 0
    step = load_step(step_file)
    # Servo angles quantise to PCA9685 ticks (~0.44 deg), so allow a few percent.
    assert step["pan_px_per_deg"] == pytest.approx(17.0, rel=0.05)
    assert step["tilt_px_per_deg"] == pytest.approx(21.0, rel=0.05)


def test_trace_visits_every_pose(tmp_path, monkeypatch, capsys):
    bs, _ = _fake_rig(monkeypatch, 18, 18)
    corners = tmp_path / "corners.json"
    save_corners(corners, SQUARE)
    cfg = Config()
    cfg.sweep.step_file = str(tmp_path / "none.json")
    args = build_parser().parse_args(["board-scan", "--crop", "0.2", "--corners", str(corners), "--trace"])
    assert bs.cmd_board_scan(args, cfg) == 0
    n = len(bs._plan_for(args, cfg)[-1])
    assert capsys.readouterr().out.count(" pan ") == n
    args = build_parser().parse_args(["board-scan", "--crop", "0.2", "--corners", str(corners), "--trace-edges"])
    assert bs.cmd_board_scan(args, cfg) == 0
    assert capsys.readouterr().out.count(" pan ") == 4


def test_max_gap_drops_a_row_when_the_leftover_is_tiny():
    # Your board: 4.64 x 3.15 squares of 8.55 x 8.56 deg.
    tp, tt = 8.55, 8.56
    w, h = 4.64 * tp, 3.15 * tt
    board = [(100 - w / 2, 90 + h / 2), (100 + w / 2, 90 + h / 2),
             (100 + w / 2, 90 - h / 2), (100 - w / 2, 90 - h / 2)]
    assert len(plan(board, (tp, tt), 0)) == 20                # cover everything: 5 x 4
    poses = plan(board, (tp, tt), 0, max_gap=0.2)
    tilts = sorted({round(p.tilt, 6) for p in poses})
    assert len(poses) == 15 and len(tilts) == 3              # 5 x 3
    assert all(d == pytest.approx(tt) for d in _spacings(tilts))   # rows butt together
    edge_gap = (tilts[0] - tt / 2) - (90 - h / 2)
    assert edge_gap == pytest.approx(0.075 * tt)             # 0.15 split over both edges
    assert edge_gap == pytest.approx((90 + h / 2) - (tilts[-1] + tt / 2))
    # Pan would leave 0.64 of a square - too much, so it keeps 5 columns.
    assert len({round(p.pan, 6) for p in poses}) == 5


def test_max_gap_never_leaves_more_than_allowed():
    from pantilt.board_scan import _fractions

    for length in np.arange(1.0, 6.0, 0.01):
        for gap in (0.0, 0.2, 0.5):
            fs = _fractions(length, 0.0, gap)
            centres = [f * length for f in fs]
            covered = (centres[-1] + 0.5) - (centres[0] - 0.5)
            assert centres[0] >= 0.5 - 1e-9 and centres[-1] <= length - 0.5 + 1e-9  # on the board
            assert length - covered <= gap + 1e-6                                   # gap respected
            assert all(b - a <= 1 + 1e-9 for a, b in zip(centres, centres[1:]))     # no inner gaps
