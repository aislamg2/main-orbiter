"""Hardware- and window-free tests for the --gui modes (no tkinter needed)."""

import pytest

import pantilt.cli as cli
import pantilt.controller as controller
import pantilt.gui as gui
from pantilt import Config, PanTilt, PCA9685
from pantilt.board_scan import load_corners
from pantilt.classify import Prediction
from pantilt.cli import build_parser
from pantilt.gui import AimSession

SQUARE = [(60, 110), (120, 110), (120, 70), (60, 70)]


class Bus:
    def __init__(self): self.regs = {0x00: 0x11}
    def write_byte_data(self, a, r, v): self.regs[r] = v
    def read_byte_data(self, a, r): return self.regs.get(r, 0)
    def write_i2c_block_data(self, a, r, d): pass
    def close(self): pass


@pytest.fixture
def pt(monkeypatch):
    monkeypatch.setattr(controller.time, "sleep", lambda s: None)
    p = PanTilt(Config(), driver=PCA9685(smbus=Bus()))
    p.center(smooth=False)
    return p


def test_moves_and_step_size(pt):
    s = AimSession(pt, teach=False, corners_file="unused.json", step=2)
    s.move(pan_dir=+1)
    s.move(tilt_dir=-1)
    assert pt.position == (92, 88)
    s.bigger_step()
    assert s.step == 5
    s.smaller_step(); s.smaller_step()
    assert s.step == 1


def test_teach_mark_undo_save(pt, tmp_path):
    f = tmp_path / "corners.json"
    s = AimSession(pt, teach=True, corners_file=f)
    for pan, tilt in SQUARE:
        pt.jump(pan=pan, tilt=tilt)
        s.mark()
    s.mark()
    assert len(s.corners) == 4 and "Already have 4" in s.msg
    s.undo()
    pt.jump(pan=60, tilt=70)
    s.mark()
    assert s.save() and s.saved
    assert load_corners(f) == [(float(p), float(t)) for p, t in SQUARE]


def test_crossed_corners_not_saved(pt, tmp_path):
    s = AimSession(pt, teach=True, corners_file=tmp_path / "c.json")
    for pan, tilt in (SQUARE[0], SQUARE[2], SQUARE[1], SQUARE[3]):
        pt.jump(pan=pan, tilt=tilt)
        s.mark()
    assert not s.save() and "not saved" in s.msg


def test_keys_mode_ignores_teach_actions(pt, tmp_path):
    s = AimSession(pt, teach=False, corners_file=tmp_path / "c.json")
    s.mark()
    assert s.corners == [] and not s.save()


def test_parser_has_gui_on_keys_teach_and_live():
    p = build_parser()
    for cmd in ("keys", "teach", "live"):
        assert p.parse_args([cmd, "--gui"]).gui
    assert not p.parse_args(["live"]).gui


class FakeLive:
    def __init__(self):
        self.latest = Prediction(label="crater", confidence=0.83, fps=24.0, infer_ms=2.2,
                                 top3=[("crater", 0.83), ("other", 0.1), ("spider", 0.04)])
        self.error = None


class FakeCamera:
    url = "http://pi:8000/"


def test_live_gui_shares_camera_and_servos_and_logs(monkeypatch, pt):
    seen = {}
    monkeypatch.setattr(gui, "run_gui", lambda cfg, **kw: seen.update(kw) or 0)
    live, cam, logged = FakeLive(), FakeCamera(), []
    args = build_parser().parse_args(["live", "--gui", "--step", "5"])
    cli._live_gui(args, Config(), pt, cam, live, lambda: logged.append(1), every=0.0, crop=0.2)

    assert seen["pt"] is pt and seen["camera"] is cam      # no second camera or servo driver
    assert seen["crop"] == 0.2 and seen["step"] == 5 and not seen["teach"]
    text = seen["extra"]()
    assert "crater" in text and "83.0%" in text and "other" in text and "pi:8000" in text
    seen["on_tick"]()
    assert logged == [1]                                   # --log rows keep coming

    live.error = RuntimeError("hailo went away")
    with pytest.raises(RuntimeError, match="hailo"):
        seen["on_tick"]()                                  # closes the window, reported


def _fake_tk(monkeypatch, frames=3):
    """Just enough of tkinter to run run_gui's loop for a few frames, no window."""
    import sys
    import types

    tk = types.ModuleType("tkinter")
    tk.TclError = RuntimeError
    shown = []

    class Widget:
        def __init__(self, *a, **k): self.photo = None
        def grid(self, *a, **k): pass
        def pack(self, *a, **k): pass
        def create_image(self, *a, **k): return 1
        def create_line(self, *a, **k): pass
        def create_rectangle(self, *a, **k): pass
        def itemconfig(self, *a, **k): pass
        def config(self, **k): shown.append(k.get("text", ""))

    class Tk(Widget):
        def __init__(self):
            super().__init__()
            self.pending, self.alive = [], True
        def title(self, t): self.t = t
        def bind(self, *a): pass
        def focus_force(self): pass
        def after(self, ms, fn): self.pending.append(fn)
        def destroy(self): self.alive = False
        def mainloop(self):
            for _ in range(frames):
                if not self.alive or not self.pending:
                    break
                self.pending.pop(0)()

    tk.Tk, tk.Canvas, tk.Frame, tk.Label, tk.Button = Tk, Widget, Widget, Widget, Widget
    tk.PhotoImage = lambda data=None: object()
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    return shown


class GrabCamera:
    url = None
    closed = False
    def grab(self):
        import numpy as np
        return np.zeros((720, 1280, 4), np.uint8)
    def close(self): GrabCamera.closed = True


def test_run_gui_reuses_callers_camera(monkeypatch, pt):
    shown = _fake_tk(monkeypatch)
    cam, ticks = GrabCamera(), []
    rc = gui.run_gui(Config(), teach=False, crop=0.2, corners_file="x.json", pt=pt, camera=cam,
                     extra=lambda: "class = crater  83.0%", on_tick=lambda: ticks.append(1))
    assert rc == 0 and len(ticks) == 4               # the first frame, then 3 from the loop
    assert "class = crater" in shown[-1] and "pan" in shown[-1]
    assert not GrabCamera.closed                     # the caller (pantilt live) still owns it


def test_run_gui_reports_a_dead_classifier(monkeypatch, pt):
    _fake_tk(monkeypatch)

    def boom():
        raise RuntimeError("classifier died")

    with pytest.raises(RuntimeError, match="classifier died"):
        gui.run_gui(Config(), teach=False, crop=0.2, corners_file="x.json", pt=pt,
                    camera=GrabCamera(), on_tick=boom)
