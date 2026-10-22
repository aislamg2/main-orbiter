"""Find + confirm, end to end on a fake rig with one fake crater on the board."""

import csv
import json
import math
import sys
import types

import numpy as np
import pytest

import pantilt.board_scan as bs
import pantilt.classify as classify
import pantilt.cli as cli
import pantilt.controller as controller
import pantilt.stream as stream
from pantilt import Config, PanTilt, PCA9685
from pantilt.board_scan import save_corners, save_step
from pantilt.classify import CLASS_NAMES
from pantilt.cli import build_parser
from pantilt.hunt import View, find_poses, pick_candidates, Candidate, windows

PPD = 17.0
CRATER = (97.0, 85.0)  # where on the board (pan, tilt) the crater is
BOARD = [(75, 100), (120, 100), (120, 72), (75, 72)]


def texture_xy(pan, tilt):
    """Top-left of the 1280 x 720 frame in the texture when aimed at (pan, tilt)."""
    return int(round(800 + (pan - 90) * PPD)), int(round(700 - (tilt - 90) * PPD))


def make_texture():
    tex = np.kron(np.random.default_rng(3).integers(0, 200, (400, 600)), np.ones((6, 6)))
    x, y = texture_xy(*CRATER)
    cx, cy = x + 640, y + 360
    yy, xx = np.ogrid[:tex.shape[0], :tex.shape[1]]
    tex[(xx - cx) ** 2 + (yy - cy) ** 2 <= 18 ** 2] = 255   # the only pure-white pixels
    return tex


def crater_score(x):
    """Fake model: confident only when the white disc is near the square's centre."""
    g = x[0, :, :, 0]
    ys, xs = np.nonzero(g == 255)
    p = np.full(len(CLASS_NAMES), 0.02, np.float32)
    if len(xs):
        off = math.hypot(xs.mean() / g.shape[1] - 0.5, ys.mean() / g.shape[0] - 0.5)
        p[CLASS_NAMES.index("crater")] = 0.95 * math.exp(-(off / 0.25) ** 2)
    p[CLASS_NAMES.index("other")] = 1 - p.sum() + 0.02
    return p


@pytest.fixture
def rig(monkeypatch, tmp_path):
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

    state = {}
    real = bs.PanTilt

    def make_pt(cfg):
        state["pt"] = real(cfg, driver=PCA9685(smbus=Bus()))
        return state["pt"]

    monkeypatch.setattr(bs, "PanTilt", make_pt)
    import pantilt.hunt as hunt
    monkeypatch.setattr(hunt, "PanTilt", make_pt)
    tex = make_texture()

    def grab():
        pan, tilt = state["pt"].position if "pt" in state else (90, 90)
        x, y = texture_xy(pan if pan is not None else 90, tilt if tilt is not None else 90)
        return tex[y:y + 720, x:x + 1280][..., None].repeat(3, 2).astype(np.uint8)

    class Cam:
        def capture_array(self, name): return grab()

    class FakeStream:
        def __init__(self, cam, status=None, **k): self.picam2 = Cam(); self.status = status
        def start(self): self.status(); return "http://pi:8000/"
        def stop(self): pass

    class FakeCamera:
        url = None
        def __init__(self, cfg, preview, port, status, thumb=None): self.status = status
        def grab(self): self.status(); return grab()
        def close(self): pass

    class FakeHailo:
        def __init__(self, hef): pass
        def __enter__(self): return self
        def __exit__(self, *e): pass
        def classify(self, x): return crater_score(x)

    monkeypatch.setattr(stream, "CameraStream", FakeStream)
    monkeypatch.setattr(cli, "_Camera", FakeCamera)
    monkeypatch.setattr(classify, "HailoClassifier", FakeHailo)

    cfg = Config()
    cfg.model.crop = 0.2
    cfg.sweep.corners_file = str(tmp_path / "corners.json")
    cfg.sweep.step_file = str(tmp_path / "step.json")
    save_corners(cfg.sweep.corners_file, BOARD)
    (tmp_path / "m.hef").write_bytes(b"")
    return cfg, tmp_path


def test_view_round_trip():
    step = {"frame": [1280, 720], "pan_px_per_deg": 17, "tilt_px_per_deg": 17,
            "pan_vec": [-17, 1.0], "tilt_vec": [0.5, 17]}
    v = View(step, 1280, 720, 100, 85)
    assert v.to_px(100, 85) == pytest.approx((640, 360))
    for x, y in ((0, 0), (1280, 720), (300, 600)):
        assert v.to_px(*v.to_angle(x, y)) == pytest.approx((x, y))
    # Against the rig's real geometry: the crater's pixel in the frame aimed at 90/90.
    rig_step = {"frame": [1280, 720], "pan_px_per_deg": PPD, "tilt_px_per_deg": PPD,
                "pan_vec": [-PPD, 0], "tilt_vec": [0, PPD]}
    v = View(rig_step, 1280, 720, 90, 90)
    x, y = texture_xy(90, 90)
    cx, cy = texture_xy(*CRATER)
    px = v.to_px(*CRATER)  # crater's pixel in the 90/90 frame = its texture pos - frame origin
    assert px == pytest.approx((cx + 640 - x, cy + 360 - y), abs=1)


def test_windows_stay_on_board():
    step = {"frame": [1280, 720], "pan_px_per_deg": PPD, "tilt_px_per_deg": PPD,
            "pan_vec": [-PPD, 0], "tilt_vec": [0, PPD]}
    v = View(step, 1280, 720, 97.5, 86)
    from pantilt.hunt import inside
    for x, y in windows(v, BOARD, 144, 48):
        for dx in (-72, 72):
            for dy in (-72, 72):
                assert inside(BOARD, *v.to_angle(x + dx, y + dy))


def test_board_fits_one_find_pose_but_big_board_needs_more():
    step = {"frame": [1280, 720], "pan_px_per_deg": PPD, "tilt_px_per_deg": PPD}
    assert len(find_poses(BOARD, step, 1280, 720, 144)) == 1
    assert len(find_poses([(30, 130), (150, 130), (150, 50), (30, 50)], step, 1280, 720, 144)) > 1


def test_nms_keeps_strongest_and_spreads_out():
    c = [Candidate(90, 90, 0.5, 0, 0, 0), Candidate(90.5, 90, 0.9, 0, 0, 0),
         Candidate(100, 90, 0.4, 0, 0, 0)]
    kept = pick_candidates(c, 2.0, 10)
    assert [k.find_p for k in kept] == [0.9, 0.4]
    assert len(pick_candidates(c, 2.0, 1)) == 1


def test_calibrate_then_hunt_photographs_the_crater(rig):
    cfg, tmp = rig
    assert bs.cmd_calibrate_step(build_parser().parse_args(["calibrate-step"]), cfg) == 0
    step = json.loads((tmp / "step.json").read_text())
    assert step["pan_vec"] == pytest.approx([-PPD, 0], abs=1)   # signs measured, not assumed
    assert step["tilt_vec"] == pytest.approx([0, PPD], abs=1)

    out = tmp / "run"
    args = build_parser().parse_args(["board-scan", "--hef", str(tmp / "m.hef"), "--out", str(out),
                                      "--threshold", "0.7", "--no-preview"])
    assert bs.cmd_board_scan(args, cfg) == 0

    cands = list(csv.DictReader((out / "candidates.csv").open()))
    saved = [c for c in cands if c["action"] == "saved"]
    assert len(saved) == 1                                   # one crater, one photo
    c = saved[0]
    assert float(c["confirm_p"]) >= 0.9 > 0.7
    assert (float(c["pan"]), float(c["tilt"])) == pytest.approx(CRATER, abs=1.0)
    photos = sorted(p.name for p in (out / "photos").iterdir())
    assert len(photos) == 2 and any(p.endswith("_crop.png") for p in photos)
    assert list(out.glob("overview_*.jpg"))
    wins = list(csv.DictReader((out / "windows.csv").open()))
    assert len(wins) > 50
    assert max(float(w["p_crater"]) for w in wins) >= cfg.sweep.look_threshold


def test_grid_mode_would_have_missed_it(rig):
    """The old grid on the same rig: the crater lands off-centre and scores low."""
    cfg, tmp = rig
    bs.cmd_calibrate_step(build_parser().parse_args(["calibrate-step"]), cfg)
    out = tmp / "grid"
    args = build_parser().parse_args(["board-scan", "--grid", "--hef", str(tmp / "m.hef"),
                                      "--out", str(out), "--threshold", "0.7", "--no-preview"])
    assert bs.cmd_board_scan(args, cfg) == 0
    rows = list(csv.DictReader((out / "poses.csv").open()))
    assert max(float(r["p_crater"]) for r in rows) < 0.3     # never even a candidate
    assert not [r for r in rows if r["action"] == "saved"]


def test_find_mode_needs_direction_calibration(rig, capsys):
    cfg, tmp = rig
    save_step(cfg.sweep.step_file, 1280, 720, PPD, PPD)  # old-style file, no vectors
    args = build_parser().parse_args(["board-scan", "--dry-run"])
    assert bs.cmd_board_scan(args, cfg) == 1
    assert "calibrate-step" in capsys.readouterr().err
