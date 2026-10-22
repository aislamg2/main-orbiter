"""Hardware-free tests for teach / board-scan: geometry, cooldown, stillness."""

import math

import numpy as np
import pytest

import pantilt.controller as controller
from pantilt import Config, PanTilt, PCA9685, load_config
from pantilt.board_scan import (
    Cooldown, center_crop, check_corners, frame_diff, load_corners, outer_poses, plan,
    save_corners, tile_deg, wait_still,
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
