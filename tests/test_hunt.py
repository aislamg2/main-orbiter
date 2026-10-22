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


def make_texture(crater=CRATER):
    tex = np.kron(np.random.default_rng(3).integers(0, 200, (400, 600)), np.ones((6, 6)))
    x, y = texture_xy(*crater)
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
def rig(request, monkeypatch, tmp_path):
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
    tex = make_texture(getattr(request, "param", CRATER))

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

    class FakePicam:
        controls: list = []
        def capture_metadata(self):
            return {"ExposureTime": 12000, "AnalogueGain": 1.5, "ColourGains": (1.8, 1.6)}
        def set_controls(self, c): FakePicam.controls.append(c)

    class FakeCamera:
        url = None
        cam = FakePicam()
        def __init__(self, cfg, preview, port, status, thumb=None): self.status = status
        def grab(self):
            json.dumps(self.status())  # what the browser polls must be valid JSON
            return grab()
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
    cfg.sweep.watch_min_s = 0.0  # the rig's frames never change; don't sit in real time
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


def test_scan_windows_form_an_even_lattice_across_stops():
    """Every board spot is within a third of a square of some window centre."""
    from pantilt.board_scan import tile_from_step
    from pantilt.hunt import local_windows, scan_poses

    step = {"frame": [1280, 720], "pan_px_per_deg": PPD, "tilt_px_per_deg": PPD,
            "pan_vec": [-PPD, 0], "tilt_vec": [0, PPD]}
    stops = scan_poses(BOARD, step, 1280, 720, 0.2, overlap=0, max_gap=0.2)
    centres = []
    for p in stops:
        v = View(step, 1280, 720, p.pan, p.tilt)
        assert len(local_windows(v, BOARD, 144, 48)) <= 9
        centres += [v.to_angle(x, y) for x, y in local_windows(v, BOARD, 144, 48)]
    tp, _ = tile_from_step(step, 1280, 720, 0.2)
    half = tp / 2
    worst = 0.0
    for pan in np.linspace(BOARD[0][0] + half, BOARD[1][0] - half, 40):
        for tilt in np.linspace(BOARD[2][1] + half, BOARD[0][1] - half, 30):
            worst = max(worst, min(math.hypot(pan - a, tilt - b) for a, b in centres))
    assert worst <= tp / 3                       # stride 1/3 square -> at most ~0.24 off-centre


def test_wide_mode_still_works(rig):
    cfg, tmp = rig
    bs.cmd_calibrate_step(build_parser().parse_args(["calibrate-step"]), cfg)
    out = tmp / "wide"
    args = build_parser().parse_args(["board-scan", "--wide", "--hef", str(tmp / "m.hef"),
                                      "--out", str(out), "--threshold", "0.7", "--no-preview"])
    assert bs.cmd_board_scan(args, cfg) == 0
    saved = [c for c in csv.DictReader((out / "candidates.csv").open()) if c["action"] == "saved"]
    assert len(saved) == 1
    assert json.loads((out / "run.json").read_text())["mode"] == "wide+confirm"


def test_scan_mode_walks_the_grid(rig):
    cfg, tmp = rig
    bs.cmd_calibrate_step(build_parser().parse_args(["calibrate-step"]), cfg)
    out = tmp / "scan"
    args = build_parser().parse_args(["board-scan", "--hef", str(tmp / "m.hef"), "--out", str(out),
                                      "--no-preview"])
    assert bs.cmd_board_scan(args, cfg) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["mode"] == "scan+revisit" and len(run["find_poses"]) > 10
    stops = {r["find_pose"] for r in csv.DictReader((out / "windows.csv").open())}
    assert len(stops) == len(run["find_poses"])  # windows scored at every stop


def test_live_view_shows_confidence(rig, monkeypatch):
    cfg, tmp = rig
    bs.cmd_calibrate_step(build_parser().parse_args(["calibrate-step"]), cfg)
    seen = []
    real = cli._Camera

    spies = []

    class Spy(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            spies.append(self)

        def grab(self):
            seen.append(self.status())
            return super().grab()

    monkeypatch.setattr(cli, "_Camera", Spy)
    args = build_parser().parse_args(["board-scan", "--hef", str(tmp / "m.hef"),
                                      "--out", str(tmp / "r"), "--no-preview"])
    assert bs.cmd_board_scan(args, cfg) == 0
    seen.append(spies[0].status())  # the browser keeps polling after the last frame
    scored = [d for d in seen if "crater %" in d]
    assert scored and {"top", "#2", "scored", "interest"} <= set(scored[-1])
    assert max(d["crater %"] for d in scored) > 80                 # the crater showed up live
    assert any(d["scored"].startswith("revisit") for d in scored)


class _Clock:
    """Each grabbed frame takes 50 ms."""
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def _watch(scores, **kw):
    """Run watch() over a sequence of crater scores, one per frame."""
    from pantilt.hunt import watch

    clock = _Clock()
    seq = iter(scores)
    crater = CLASS_NAMES.index("crater")

    def grab():
        clock.t += 0.05
        return next(seq)

    def score(v):
        p = np.zeros((1, len(CLASS_NAMES)), np.float32)
        p[0, crater] = v
        return p

    args = dict(smooth=5, min_s=0.8, max_s=3.0, rise_level=0.1, stop_at=0.7)
    args.update(kw)
    return watch(grab, score, crater, clock=clock, **args)


def test_watch_moves_on_quickly_from_empty_board():
    w = _watch([0.01] * 200)
    assert w.seconds == pytest.approx(0.8, abs=0.06)


def test_watch_stays_while_the_score_climbs():
    # Exposure settling: the crater score creeps up from 0.2 to 0.65 over 2 s.
    rising = list(np.linspace(0.2, 0.65, 40)) + [0.65] * 200
    w = _watch(rising)
    assert w.seconds > 2.0 and w.best > 0.6            # waited it out
    short = _watch(rising, max_s=0.8)
    assert short.best < 0.4 and short.rising           # a fixed short look misses it


def test_watch_stops_once_good_enough():
    w = _watch(list(np.linspace(0.2, 1.0, 60)) + [1.0] * 100)
    assert w.best >= 0.7 and w.seconds < 2.5


def test_watch_ignores_one_lucky_frame():
    w = _watch([0.05] * 10 + [0.95] + [0.05] * 200)
    assert w.best < 0.3                                 # 5-frame average


def test_watch_gives_up_on_a_flat_maybe():
    w = _watch([0.4] * 200)                             # worth a look, never improves
    assert w.seconds < 1.8 and w.best == pytest.approx(0.4)


def test_lock_exposure(rig):
    cfg, tmp = rig
    bs.cmd_calibrate_step(build_parser().parse_args(["calibrate-step"]), cfg)
    cam = cli._Camera.cam
    cam.controls.clear()
    out = tmp / "locked"
    args = build_parser().parse_args(["board-scan", "--lock-exposure", "--hef", str(tmp / "m.hef"),
                                      "--out", str(out), "--no-preview"])
    assert bs.cmd_board_scan(args, cfg) == 0
    assert cam.controls == [{"AeEnable": False, "AwbEnable": False, "ExposureTime": 12000,
                             "AnalogueGain": 1.5, "ColourGains": (1.8, 1.6)}]
    assert json.loads((out / "run.json").read_text())["exposure_locked"]["ExposureTime"] == 12000
    # Without the flag nothing is locked.
    cam.controls.clear()
    args = build_parser().parse_args(["board-scan", "--hef", str(tmp / "m.hef"),
                                      "--out", str(tmp / "free"), "--no-preview"])
    assert bs.cmd_board_scan(args, cfg) == 0 and cam.controls == []


def test_wide_header_says_wide(rig, capsys):
    cfg, tmp = rig
    bs.cmd_calibrate_step(build_parser().parse_args(["calibrate-step"]), cfg)
    capsys.readouterr()
    assert bs.cmd_board_scan(build_parser().parse_args(["board-scan", "--wide", "--dry-run"]), cfg) == 0
    assert "mode         wide view + revisit" in capsys.readouterr().out


# Just inside the taught outline's top-left corner: 0.15 of a square from each edge.
EDGE_CRATER = (BOARD[0][0] + 0.15 * 8.47, BOARD[0][1] - 0.15 * 8.47)


@pytest.mark.parametrize("rig", [EDGE_CRATER], indirect=True)
@pytest.mark.parametrize("margin, photos", [(0.0, 0), (0.4, 1)])
def test_crater_near_the_outline_needs_edge_margin(rig, margin, photos):
    cfg, tmp = rig
    bs.cmd_calibrate_step(build_parser().parse_args(["calibrate-step"]), cfg)
    cfg.sweep.edge_margin = margin
    out = tmp / f"edge{margin}"
    args = build_parser().parse_args(["board-scan", "--hef", str(tmp / "m.hef"), "--out", str(out),
                                      "--threshold", "0.7", "--no-preview"])
    assert bs.cmd_board_scan(args, cfg) == 0
    saved = [c for c in csv.DictReader((out / "candidates.csv").open()) if c["action"] == "saved"]
    assert len(saved) == photos
    if photos:
        assert (float(saved[0]["pan"]), float(saved[0]["tilt"])) == pytest.approx(EDGE_CRATER, abs=1.0)
