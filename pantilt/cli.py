"""Command-line interface:  python -m pantilt <command>   (or `pantilt <command>`)."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import load_config
from .controller import PanTilt
from .scan import raster
from .board_scan import cmd_board_scan, cmd_calibrate_step, cmd_teach
from .clean import cmd_clean
from .survey import cmd_measure_k, cmd_survey


# -- commands ----------------------------------------------------------------
def cmd_check(args, cfg) -> int:
    """Hardware sanity check: I2C bus, PCA9685, and the AI HAT+ (Hailo)."""
    ok = True
    dev = Path(f"/dev/i2c-{cfg.driver.i2c_bus}")
    if dev.exists():
        print(f"[ok]   {dev} exists")
    else:
        print(f"[FAIL] {dev} missing - enable I2C: sudo raspi-config nonint do_i2c 0, then reboot")
        return 1

    from .pca9685 import PCA9685

    try:
        drv = PCA9685(cfg.driver.i2c_bus, cfg.driver.address, cfg.driver.oscillator_hz)
        drv.ping()
        drv.close()
        print(f"[ok]   PCA9685 answering at 0x{cfg.driver.address:02X}")
    except OSError as e:
        ok = False
        print(f"[FAIL] no PCA9685 at 0x{cfg.driver.address:02X} ({e}) - check SDA=pin 3, SCL=pin 5, VCC, GND")

    if shutil.which("lspci"):
        out = subprocess.run(["lspci"], capture_output=True, text=True).stdout
        if "hailo" in out.lower():
            print("[ok]   Hailo accelerator (AI HAT+) found on PCIe")
        else:
            print("[warn] Hailo not found on PCIe - reseat the AI HAT+ ribbon cable if you need it")
    print("\nReminder: servo power comes from the driver's V+ terminal (external 5-6 V), not the Pi.")
    return 0 if ok else 1


def cmd_center(args, cfg) -> int:
    with PanTilt(cfg) as pt:
        print("pan=%.0f tilt=%.0f" % pt.center(smooth=False))
    return 0


def cmd_goto(args, cfg) -> int:
    with PanTilt(cfg) as pt:
        print("pan=%.0f tilt=%.0f" % pt.jump(pan=args.pan, tilt=args.tilt))
    return 0


def cmd_off(args, cfg) -> int:
    with PanTilt(cfg) as pt:
        pt.release()
    print("servos released")
    return 0


def cmd_sweep(args, cfg) -> int:
    """Move each axis center -> min -> max -> center to check range and direction."""
    with PanTilt(cfg) as pt:
        pt.center(smooth=False)
        time.sleep(0.5)
        axes = ["pan", "tilt"] if args.axis == "both" else [args.axis]
        for name in axes:
            sc = getattr(cfg, name)
            for target in (sc.min_angle, sc.max_angle, sc.center):
                print(f"{name} -> {target:.0f}")
                pt.move_to(**{name: target})
                time.sleep(0.5)
        if args.release:
            pt.release()
    return 0


def cmd_keys(args, cfg) -> int:
    if args.gui:
        from .gui import run_gui
        return run_gui(cfg, teach=False, crop=cfg.model.crop, corners_file=cfg.sweep.corners_file, step=args.step)

    import curses

    def loop(scr):
        scr.keypad(True)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        step = args.step
        with PanTilt(cfg) as pt:
            pt.center(smooth=False)
            msg = ""
            while True:
                p, t = pt.position
                scr.erase()
                scr.addstr(0, 0, "Pan-tilt keyboard control")
                scr.addstr(2, 0, f"pan  = {p:6.1f}   ({cfg.pan.min_angle:.0f}-{cfg.pan.max_angle:.0f})")
                scr.addstr(3, 0, f"tilt = {t:6.1f}   ({cfg.tilt.min_angle:.0f}-{cfg.tilt.max_angle:.0f})")
                scr.addstr(4, 0, f"step = {step} deg")
                scr.addstr(6, 0, "arrows: move   +/-: step size   c: center")
                scr.addstr(7, 0, "r: release (go limp)   q: quit (servos hold position)")
                scr.addstr(9, 0, msg)
                scr.refresh()
                k = scr.getch()
                msg = ""
                if k == curses.KEY_UP:
                    pt.nudge(dtilt=+step)
                elif k == curses.KEY_DOWN:
                    pt.nudge(dtilt=-step)
                elif k == curses.KEY_RIGHT:
                    pt.nudge(dpan=+step)
                elif k == curses.KEY_LEFT:
                    pt.nudge(dpan=-step)
                elif k in (ord("+"), ord("=")):
                    step = min(step + 1, 20)
                elif k in (ord("-"), ord("_")):
                    step = max(step - 1, 1)
                elif k in (ord("c"), ord("C")):
                    pt.center()
                elif k in (ord("r"), ord("R")):
                    pt.release()
                    msg = "released - next key press re-engages"
                elif k in (ord("q"), ord("Q"), 27):
                    break

    curses.wrapper(loop)
    return 0


def cmd_scan(args, cfg) -> int:
    """Raster-scan the camera over a grid, optionally saving a photo at each pose."""
    pan_range = tuple(args.pan) if args.pan else (cfg.pan.min_angle, cfg.pan.max_angle)
    tilt_range = tuple(args.tilt) if args.tilt else (cfg.tilt.min_angle, cfg.tilt.max_angle)
    poses = list(raster(pan_range, tilt_range, args.cols, args.rows))

    cam = writer = None
    if args.capture:
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("picamera2 not found. Install it with `sudo apt install python3-picamera2` and "
                  "create the venv with --system-site-packages.", file=sys.stderr)
            return 1
        out = Path(args.capture)
        out.mkdir(parents=True, exist_ok=True)
        from .stream import camera_transform

        cam = Picamera2()
        cam.configure(cam.create_still_configuration(transform=camera_transform(cfg.camera)))
        cam.start()
        time.sleep(1)  # let auto-exposure settle
        manifest = (out / "poses.csv").open("w", newline="")
        writer = csv.writer(manifest)
        writer.writerow(["index", "pan", "tilt", "file"])

    try:
        with PanTilt(cfg) as pt:
            for i, (pan, tilt) in enumerate(poses):
                pt.move_to(pan=pan, tilt=tilt, settle=True)
                line = f"[{i + 1}/{len(poses)}] pan={pan:.1f} tilt={tilt:.1f}"
                if cam:
                    name = f"{i:03d}_p{pan:05.1f}_t{tilt:05.1f}.jpg"
                    cam.capture_file(str(out / name))
                    writer.writerow([i, f"{pan:.1f}", f"{tilt:.1f}", name])
                    line += f" -> {name}"
                else:
                    time.sleep(args.dwell)
                print(line)
            pt.center()
    finally:
        if cam:
            cam.stop()
            manifest.close()
    return 0


# -- live classification -------------------------------------------------------
class _Camera:
    """The camera, with or without the browser preview. grab() returns the newest frame."""

    def __init__(self, cfg, preview: bool, port: int | None, status, thumb=None):
        self.stream = self.picam2 = None
        if preview:
            from .stream import CameraStream

            self.stream = CameraStream(cfg.camera, status=status, port=port, thumb=thumb)
            self.url = self.stream.start()
            self.cam = self.stream.picam2
        else:
            from picamera2 import Picamera2

            from .stream import camera_transform

            self.picam2 = Picamera2()
            from .stream import FULL_ARRAY

            self.picam2.configure(self.picam2.create_video_configuration(
                main={"size": (cfg.camera.preview_width, cfg.camera.preview_height)},
                raw={"size": FULL_ARRAY},
                transform=camera_transform(cfg.camera),
            ))
            self.picam2.start()
            self.url = None
            self.cam = self.picam2

    def grab(self):
        return self.cam.capture_array("main")

    def close(self):
        if self.stream:
            self.stream.stop()
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()


def _thumb_source(cam_ref: dict, crop: float):
    """Returns thumb(): JPEG of the exact array handed to the Hailo, at most 4 per second.

    Cached behind a lock so a browser polling the endpoint cannot stack up
    camera grabs on the capture loop. cam_ref["camera"] is filled in once the
    camera is open.
    """
    from .classify import preprocess

    _thumb_cache: dict = {"at": 0.0, "jpg": None}
    _thumb_lock = threading.Lock()

    def thumb():
        import io

        now = time.time()
        if now - _thumb_cache["at"] < 0.25:
            return _thumb_cache["jpg"]
        with _thumb_lock:
            if time.time() - _thumb_cache["at"] < 0.25:
                return _thumb_cache["jpg"]
            cam = cam_ref.get("camera")
            if cam is None:
                return None
            try:
                from PIL import Image

                arr = preprocess(cam.grab(), crop=crop)[0]
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="JPEG", quality=85)
                _thumb_cache["jpg"] = buf.getvalue()
            except Exception:
                _thumb_cache["jpg"] = None
            _thumb_cache["at"] = time.time()
            return _thumb_cache["jpg"]

    return thumb


def cmd_live(args, cfg) -> int:
    """Real-time landmark classification on the Hailo-8, with pan-tilt control."""
    from .classify import HailoClassifier, LiveClassifier

    m = cfg.model
    hef = Path(args.hef or m.hef_path)
    if not hef.exists():
        print(f"model not found: {hef}\nCopy mars_model_float32.hef from mars-landmark-classification "
              f"(hailo-inference branch) to {m.hef_path}, or pass --hef PATH.", file=sys.stderr)
        return 1
    try:
        import picamera2  # noqa: F401
        import hailo_platform  # noqa: F401
    except ImportError as e:
        print(f"{e.name} not found. Install with `sudo apt install python3-picamera2 python3-hailort` "
              "and create the venv with --system-site-packages.", file=sys.stderr)
        return 1

    crop = args.crop if args.crop is not None else m.crop
    pt_ref: dict = {}
    cam_ref: dict = {}
    thumb = _thumb_source(cam_ref, crop)

    def status():
        d = dict(live.status()) if live else {}
        d["crop"] = f"{crop:.2f}"
        pt = pt_ref.get("pt")
        if pt:
            p, t = pt.position
            d["pan"] = p if p is not None else "-"
            d["tilt"] = t if t is not None else "-"
        return d

    live = None
    log = writer = None
    camera = None
    try:
        with HailoClassifier(str(hef)) as clf, PanTilt(cfg) as pt:
            pt_ref["pt"] = pt
            camera = _Camera(cfg, preview=not args.no_preview, port=args.port,
                             status=status, thumb=thumb)
            cam_ref["camera"] = camera
            time.sleep(1)  # let auto-exposure settle
            live = LiveClassifier(camera.grab, clf.classify, smooth=m.smooth_frames, crop=crop,
                                  min_confidence=m.min_confidence)
            if args.log:
                Path(args.log).parent.mkdir(parents=True, exist_ok=True)
                log = open(args.log, "w", newline="")
                writer = csv.writer(log)
                writer.writerow(["time", "pan", "tilt", "label", "confidence", "infer_ms",
                                 *[f"p_{n.replace(' ', '_')}" for n in _class_names()]])

            def record():
                p = live.latest
                if writer and p.probs is not None:
                    pan, tilt = pt.position
                    writer.writerow([f"{time.time():.3f}", _fmt(pan), _fmt(tilt), p.label,
                                     f"{p.confidence:.4f}", f"{p.infer_ms:.2f}",
                                     *[f"{v:.4f}" for v in p.probs]])
                    log.flush()

            if camera.url:
                print(f"live view: {camera.url}")
            live.start()

            try:
                if args.scan:
                    _live_scan(args, cfg, pt, live, record)
                elif args.no_keys:
                    _live_print(live, record, m.log_every_s)
                else:
                    _live_keys(args, cfg, pt, live, record, m.log_every_s, camera.url)
            finally:
                # Must happen while the pipeline is still open: if the `with`
                # closes it first, the thread's next infer() hits a None and
                # that AttributeError masks however the run really ended.
                live.stop()
    finally:
        if live:
            live.stop()
        if camera:
            camera.close()
        if log:
            log.close()
    if live and live.error:
        raise live.error
    return 0


def _class_names():
    from .classify import CLASS_NAMES

    return CLASS_NAMES


def _fmt(v):
    return "" if v is None else f"{v:.1f}"


def _check(live):
    if live.error:
        raise live.error


def _live_print(live, record, every):
    """Headless: print one line per interval until Ctrl+C."""
    while True:
        time.sleep(every)
        _check(live)
        p = live.latest
        print(f"{p.label:<14} {p.confidence * 100:5.1f}%   {p.fps:5.1f} fps   hailo {p.infer_ms:.1f} ms")
        record()


def _live_scan(args, cfg, pt, live, record):
    """Raster-scan; at each pose wait for fresh frames, then log the prediction."""
    pan_range = tuple(args.pan) if args.pan else (cfg.pan.min_angle, cfg.pan.max_angle)
    tilt_range = tuple(args.tilt) if args.tilt else (cfg.tilt.min_angle, cfg.tilt.max_angle)
    poses = list(raster(pan_range, tilt_range, args.cols, args.rows))
    for i, (pan, tilt) in enumerate(poses):
        pt.move_to(pan=pan, tilt=tilt, settle=True)
        start = live.latest.frames
        # Wait until the smoothing window holds only frames taken at this pose.
        while live.latest.frames - start < cfg.model.smooth_frames + 1:
            _check(live)
            time.sleep(0.01)
        p = live.latest
        record()
        top = ", ".join(f"{n} {v * 100:.0f}%" for n, v in p.top3)
        print(f"[{i + 1}/{len(poses)}] pan={pan:.1f} tilt={tilt:.1f} -> {top}")
    pt.center()


def _live_keys(args, cfg, pt, live, record, every, url):
    import curses

    def loop(scr):
        scr.keypad(True)
        scr.timeout(100)  # refresh the readout even without key presses
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        step = args.step
        pt.center(smooth=False)
        last_log = 0.0
        msg = ""
        while True:
            _check(live)
            now = time.time()
            if now - last_log >= every:
                record()
                last_log = now
            p = live.latest
            pan, tilt = pt.position
            scr.erase()
            rows = [
                "Live landmark classification (Hailo-8)",
                f"view: {url}" if url else "view: (preview off)",
                "",
                f"class = {p.label}   {p.confidence * 100:5.1f}%",
                *[f"   {n:<14} {v * 100:5.1f}%" for n, v in p.top3[1:]],
                "",
                f"fps   = {p.fps:5.1f}    hailo = {p.infer_ms:5.1f} ms",
                f"pan   = {_fmt(pan):>5}    tilt  = {_fmt(tilt):>5}    step = {step} deg",
                "",
                "arrows: move   +/-: step   c: center   r: release   q: quit",
                msg,
            ]
            for y, text in enumerate(rows):
                try:
                    scr.addstr(y, 0, text)
                except curses.error:
                    pass  # terminal too small
            scr.refresh()
            k = scr.getch()
            if k == -1:
                continue
            msg = ""
            if k == curses.KEY_UP:
                pt.nudge(dtilt=+step)
            elif k == curses.KEY_DOWN:
                pt.nudge(dtilt=-step)
            elif k == curses.KEY_RIGHT:
                pt.nudge(dpan=+step)
            elif k == curses.KEY_LEFT:
                pt.nudge(dpan=-step)
            elif k in (ord("+"), ord("=")):
                step = min(step + 1, 20)
            elif k in (ord("-"), ord("_")):
                step = max(step - 1, 1)
            elif k in (ord("c"), ord("C")):
                pt.center()
            elif k in (ord("r"), ord("R")):
                pt.release()
                msg = "released - next arrow key re-engages"
            elif k in (ord("q"), ord("Q"), 27):
                break

    curses.wrapper(loop)


# -- entry point -------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pantilt", description=__doc__)
    p.add_argument("--config", help="path to config.toml (default: ./config.toml if present)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="check I2C, the PCA9685, and the AI HAT+").set_defaults(func=cmd_check)
    sub.add_parser("center", help="jump both servos to center").set_defaults(func=cmd_center)
    sub.add_parser("off", help="release both servos (go limp)").set_defaults(func=cmd_off)

    g = sub.add_parser("goto", help="jump to an absolute pan/tilt angle")
    g.add_argument("pan", type=float)
    g.add_argument("tilt", type=float)
    g.set_defaults(func=cmd_goto)

    s = sub.add_parser("sweep", help="test each axis through its full range")
    s.add_argument("--axis", choices=["pan", "tilt", "both"], default="both")
    s.add_argument("--release", action="store_true", help="go limp when finished")
    s.set_defaults(func=cmd_sweep)

    k = sub.add_parser("keys", help="drive the camera with the arrow keys")
    k.add_argument("--step", type=int, default=2, help="degrees per key press (default 2)")
    k.add_argument("--gui", action="store_true", help="open a window with the live camera instead of the terminal screen")
    k.set_defaults(func=cmd_keys)

    sc = sub.add_parser("scan", help="raster-scan a grid of poses, optionally capturing photos")
    sc.add_argument("--pan", nargs=2, type=float, metavar=("MIN", "MAX"))
    sc.add_argument("--tilt", nargs=2, type=float, metavar=("MIN", "MAX"))
    sc.add_argument("--cols", type=int, default=5, help="poses per row (default 5)")
    sc.add_argument("--rows", type=int, default=3, help="number of rows (default 3)")
    sc.add_argument("--capture", metavar="DIR", help="save a photo at each pose into DIR")
    sc.add_argument("--dwell", type=float, default=0.5, help="pause per pose without --capture")
    sc.set_defaults(func=cmd_scan)

    mk = sub.add_parser("measure-k", help="measure mm per pixel from two marks on the board")
    mk.add_argument("--separation", type=float, default=1000.0, help="distance between the marks in mm (default 1000)")
    mk.add_argument("--out", default="k_measure.png", help="where to save the frame")
    mk.set_defaults(func=cmd_measure_k)

    sv = sub.add_parser("survey", help="scan the board, classify each tile, save the craters")
    sv.add_argument("--dry-run", action="store_true", help="print the plan, move nothing")
    sv.add_argument("--threshold", type=float, default=0.5, help="crater probability that counts as a detection (default 0.5)")
    sv.add_argument("--crop", type=float, help="fraction of the frame to classify (default: [model] crop)")
    sv.add_argument("--repeat", type=int, default=1, help="captures averaged per pose to cut sensor noise (default 1)")
    sv.add_argument("--save-all", action="store_true", help="save every tile, not just the hits (needed for a mosaic)")
    sv.add_argument("--serpentine", action="store_true", help="halve the travel, at the cost of backlash flipping sign per row")
    sv.add_argument("--hef", help="model override (default: [model] hef_path)")
    sv.add_argument("--out", metavar="DIR", help="run directory (default: data/runs/<timestamp>)")
    sv.set_defaults(func=cmd_survey)

    te = sub.add_parser("teach", help="aim at the 4 board corners and save them for board-scan")
    te.add_argument("--crop", type=float, help="red box size in the live view (default: [model] crop)")
    te.add_argument("--file", help="where to save (default: [sweep] corners_file)")
    te.add_argument("--gui", action="store_true", help="open a window with the live camera instead of the terminal screen")
    te.set_defaults(func=cmd_teach)

    cs = sub.add_parser("calibrate-step", help="measure the servo step that moves the image one crop square")
    cs.add_argument("--crop", type=float, help="crop to report the result for (default: [model] crop)")
    cs.add_argument("--delta", type=float, default=5.0, help="test move in degrees (default 5)")
    cs.add_argument("--reps", type=int, default=3, help="measurements per axis (default 3)")
    cs.add_argument("--file", help="where to save (default: [sweep] step_file)")
    cs.set_defaults(func=cmd_calibrate_step)

    bs = sub.add_parser("board-scan", help="scan the taught board, go back to likely craters, photograph the real ones")
    bs.add_argument("--crop", type=float, help="fraction of the frame to classify (default: [model] crop)")
    bs.add_argument("--threshold", type=float, help="target probability that triggers a photo (default: [sweep] threshold)")
    bs.add_argument("--cooldown-tiles", type=float, help="no photo within this many crop widths of a saved one")
    bs.add_argument("--cooldown-s", type=float, help="minimum seconds between photos")
    bs.add_argument("--max-photos", type=int, help="stop saving after this many (0 = no limit)")
    bs.add_argument("--speed", type=float, help="degrees per second between poses (default: [sweep] speed_dps)")
    bs.add_argument("--passes", type=int, default=1, help="times over the board (0 = until Ctrl+C)")
    bs.add_argument("--grid", action="store_true", help="old one-look-per-square grid, no revisits")
    bs.add_argument("--wide", action="store_true", help="find from one frame of the whole board instead of walking the grid")
    bs.add_argument("--lock-exposure", action="store_true", help="freeze exposure/white balance at the start of the run")
    bs.add_argument("--one-way", action="store_true", help="(--grid) every row left to right instead of serpentine")
    bs.add_argument("--dry-run", action="store_true", help="print the grid, move nothing")
    bs.add_argument("--trace", action="store_true", help="walk every grid point without classifying, to check coverage")
    bs.add_argument("--trace-edges", action="store_true", help="visit only the 4 outermost grid points, to check the board edge")
    bs.add_argument("--hold", type=float, help="seconds to pause at each trace stop (default 1, or 3 with --trace-edges)")
    bs.add_argument("--corners", help="corners file (default: [sweep] corners_file)")
    bs.add_argument("--hef", help="model override (default: [model] hef_path)")
    bs.add_argument("--out", metavar="DIR", help="run directory (default: data/runs/board-<timestamp>)")
    bs.add_argument("--no-preview", action="store_true", help="don't serve the browser live view")
    bs.add_argument("--port", type=int, help="live view port (default: preview_port)")
    bs.set_defaults(func=cmd_board_scan)

    cl = sub.add_parser("clean", help="delete saved photos from data/runs (asks first)")
    which = cl.add_mutually_exclusive_group()
    which.add_argument("--all", action="store_true", help="every run")
    which.add_argument("--keep", type=int, metavar="N", help="all but the newest N runs")
    which.add_argument("--run", action="append", metavar="NAME", help="this run (repeatable)")
    cl.add_argument("--everything", action="store_true", help="remove whole run folders, CSVs included")
    cl.add_argument("--yes", "-y", action="store_true", help="don't ask for confirmation")
    cl.add_argument("--runs-dir", default="data/runs", help="where runs are saved (default: data/runs)")
    cl.set_defaults(func=cmd_clean)

    lv = sub.add_parser("live", help="real-time landmark classification on the Hailo-8")
    lv.add_argument("--hef", help="compiled model (default: [model] hef_path in config.toml)")
    lv.add_argument("--crop", type=float, help="fraction of the frame to classify, centered (0-1]")
    lv.add_argument("--log", metavar="CSV", help="save predictions with pan/tilt to a CSV file")
    lv.add_argument("--no-preview", action="store_true", help="don't serve the browser live view")
    lv.add_argument("--port", type=int, help="live view port (default: preview_port)")
    lv.add_argument("--no-keys", action="store_true", help="no keyboard control; print predictions")
    lv.add_argument("--step", type=int, default=2, help="degrees per key press (default 2)")
    lv.add_argument("--scan", action="store_true", help="raster-scan and classify at each pose")
    lv.add_argument("--pan", nargs=2, type=float, metavar=("MIN", "MAX"), help="scan pan range")
    lv.add_argument("--tilt", nargs=2, type=float, metavar=("MIN", "MAX"), help="scan tilt range")
    lv.add_argument("--cols", type=int, default=5, help="scan poses per row (default 5)")
    lv.add_argument("--rows", type=int, default=3, help="scan rows (default 3)")
    lv.set_defaults(func=cmd_live)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        print("\nstopped")
        return 130
    except PermissionError:
        print("permission denied on the I2C bus - add yourself to the i2c group: "
              "sudo usermod -aG i2c $USER (then log out and back in)", file=sys.stderr)
        return 1
