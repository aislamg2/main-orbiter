"""Command-line interface:  python -m pantilt <command>   (or `pantilt <command>`)."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import load_config
from .controller import PanTilt
from .scan import raster


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
        cam = Picamera2()
        cam.configure(cam.create_still_configuration())
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
    k.set_defaults(func=cmd_keys)

    sc = sub.add_parser("scan", help="raster-scan a grid of poses, optionally capturing photos")
    sc.add_argument("--pan", nargs=2, type=float, metavar=("MIN", "MAX"))
    sc.add_argument("--tilt", nargs=2, type=float, metavar=("MIN", "MAX"))
    sc.add_argument("--cols", type=int, default=5, help="poses per row (default 5)")
    sc.add_argument("--rows", type=int, default=3, help="number of rows (default 3)")
    sc.add_argument("--capture", metavar="DIR", help="save a photo at each pose into DIR")
    sc.add_argument("--dwell", type=float, default=0.5, help="pause per pose without --capture")
    sc.set_defaults(func=cmd_scan)
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
