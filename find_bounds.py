#!/usr/bin/env python3
"""find_bounds.py - drive the pan-tilt freely with the arrow keys and watch the live angles.

There are NO software limits here: each servo can go anywhere in its full
0-180 degree range. Watch the bracket, and if a servo buzzes, strains or the
bracket hits something, back off (or press r to let the servos go limp).

Channels, direction (invert) and pulse widths still come from config.toml;
only the min/max angle limits are ignored.

Run from the repo folder:
    python find_bounds.py
    python find_bounds.py --preview          # + live camera in your browser
    python find_bounds.py --step 5 --start-pan 90 --start-tilt 90

With --preview, open the printed address (http://<pi-ip>:8000/) in a browser
on your Mac or the Pi. It shows the camera with a center crosshair and the live
angles. The image is flipped 180 degrees when [camera] rotate_180 = true in
config.toml.

Keys:
    arrow keys   move (up/down = tilt, left/right = pan)
    + / -        bigger / smaller step
    m            mark the current position (saved to marks.csv)
    c            back to center
    r            release (servos go limp; any arrow key re-engages)
    q            quit and print a summary
"""

from __future__ import annotations

import argparse
import csv
import curses
import dataclasses
from datetime import datetime
from pathlib import Path

from pantilt import load_config
from pantilt.controller import PanTilt

FULL_MIN, FULL_MAX = 0.0, 180.0
STEPS = [0.5, 1, 2, 5, 10, 20]


class Explorer:
    def __init__(self, pt: PanTilt, step: float, marks_file: Path | None):
        self.pt = pt
        self.step_i = min(range(len(STEPS)), key=lambda i: abs(STEPS[i] - step))
        self.marks: list[tuple[float, float]] = []
        self.marks_file = marks_file
        self.seen = {"pan": [None, None], "tilt": [None, None]}
        self.released = False
        self.msg = ""
        self.preview_url: str | None = None

    def status(self) -> dict:
        """Live values shown on the browser preview."""
        p, t = self.pt.position
        return {
            "pan": p,
            "tilt": t,
            "step": self.step,
            "servos": "released" if self.released else "holding",
            "marks": len(self.marks),
        }

    @property
    def step(self) -> float:
        return STEPS[self.step_i]

    def _track(self):
        for name, angle in zip(("pan", "tilt"), self.pt.position):
            lo, hi = self.seen[name]
            self.seen[name] = [angle if lo is None else min(lo, angle),
                               angle if hi is None else max(hi, angle)]

    def move(self, dpan: float = 0, dtilt: float = 0):
        p, t = self.pt.position
        new_p, new_t = p + dpan, t + dtilt
        if not (FULL_MIN <= new_p <= FULL_MAX and FULL_MIN <= new_t <= FULL_MAX):
            self.msg = "at the servo's full 0-180 travel - can't go further"
        self.pt.jump(pan=new_p, tilt=new_t)
        self.released = False
        self._track()

    def mark(self):
        p, t = self.pt.position
        self.marks.append((p, t))
        self.msg = f"mark {len(self.marks)} saved: pan={p:.1f} tilt={t:.1f}"
        if self.marks_file:
            new = not self.marks_file.exists()
            with self.marks_file.open("a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "pan", "tilt"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), f"{p:.1f}", f"{t:.1f}"])

    def handle_key(self, k: int) -> bool:
        """Apply one key press. Returns False when it's time to quit."""
        self.msg = ""
        s = self.step
        if k == curses.KEY_UP:
            self.move(dtilt=+s)
        elif k == curses.KEY_DOWN:
            self.move(dtilt=-s)
        elif k == curses.KEY_RIGHT:
            self.move(dpan=+s)
        elif k == curses.KEY_LEFT:
            self.move(dpan=-s)
        elif k in (ord("+"), ord("=")):
            self.step_i = min(self.step_i + 1, len(STEPS) - 1)
        elif k in (ord("-"), ord("_")):
            self.step_i = max(self.step_i - 1, 0)
        elif k in (ord("m"), ord("M")):
            self.mark()
        elif k in (ord("c"), ord("C")):
            self.pt.jump(pan=self.pt.config.pan.center, tilt=self.pt.config.tilt.center)
            self.released = False
            self._track()
        elif k in (ord("r"), ord("R")):
            self.pt.release()
            self.released = True
            self.msg = "released - servos limp. Press an arrow key to re-engage."
        elif k in (ord("q"), ord("Q"), 27):
            return False
        return True

    def lines(self) -> list[str]:
        p, t = self.pt.position
        pp, tp = self.pt.pan.pulse_for(p), self.pt.tilt.pulse_for(t)
        (plo, phi), (tlo, thi) = self.seen["pan"], self.seen["tilt"]
        state = "RELEASED (limp)" if self.released else "holding"
        out = [
            "Pan-tilt bounds finder  (no limits - watch the bracket!)",
            "",
            f"  PAN   {p:6.1f} deg    pulse {pp:6.0f} us    visited {plo:5.1f} .. {phi:5.1f}",
            f"  TILT  {t:6.1f} deg    pulse {tp:6.0f} us    visited {tlo:5.1f} .. {thi:5.1f}",
            "",
            f"  step {self.step} deg    servos {state}    marks {len(self.marks)}",
            "",
            "  arrows move   +/- step   m mark   c center   r release   q quit",
            "",
        ]
        if self.preview_url:
            out[1:1] = [f"  Live camera: {self.preview_url}", ""]
        for i, (mp, mt) in enumerate(self.marks[-8:], start=max(1, len(self.marks) - 7)):
            out.append(f"  mark {i}: pan={mp:.1f}  tilt={mt:.1f}")
        if self.msg:
            out += ["", "  " + self.msg]
        return out

    def summary(self) -> str:
        (plo, phi), (tlo, thi) = self.seen["pan"], self.seen["tilt"]
        out = [
            "",
            "=== Session summary ===",
            f"pan  visited {plo:.1f} .. {phi:.1f} deg",
            f"tilt visited {tlo:.1f} .. {thi:.1f} deg",
        ]
        if self.marks:
            out.append("")
            out.append("Marks:")
            for i, (mp, mt) in enumerate(self.marks, 1):
                out.append(f"  {i}: pan={mp:.1f}  tilt={mt:.1f}")
            ps = [m[0] for m in self.marks]
            ts = [m[1] for m in self.marks]
            if min(ps) < max(ps) and min(ts) < max(ts):
                out += [
                    "",
                    "If your marks are the safe edges, these go in config.toml:",
                    f"  [pan]   min_angle = {min(ps):.0f}   max_angle = {max(ps):.0f}",
                    f"  [tilt]  min_angle = {min(ts):.0f}   max_angle = {max(ts):.0f}",
                    f"  scan:   pantilt scan --pan {min(ps):.0f} {max(ps):.0f} --tilt {min(ts):.0f} {max(ts):.0f}",
                ]
            else:
                out.append("\n(Mark opposite corners of the area you want to get suggested limits.)")
            if self.marks_file:
                out.append(f"\nMarks saved to {self.marks_file}")
        out.append("Servos are still holding position. Run `pantilt off` to release them.")
        return "\n".join(out)


def run(scr, explorer: Explorer):
    scr.keypad(True)
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    while True:
        scr.erase()
        for row, text in enumerate(explorer.lines()):
            try:
                scr.addstr(row, 0, text)
            except curses.error:
                pass  # terminal too small for this line
        scr.refresh()
        if not explorer.handle_key(scr.getch()):
            return


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Drive the pan-tilt with no limits and read live angles.")
    ap.add_argument("--config", help="config.toml to read channels/invert/pulses from")
    ap.add_argument("--step", type=float, default=2, help="starting step in degrees (default 2)")
    ap.add_argument("--start-pan", type=float, help="starting pan angle (default: config center)")
    ap.add_argument("--start-tilt", type=float, help="starting tilt angle (default: config center)")
    ap.add_argument("--marks", default="marks.csv", help="file to save marks to ('' to disable)")
    ap.add_argument("--preview", action="store_true", help="stream the live camera to a web page")
    ap.add_argument("--port", type=int, help="web preview port (default: [camera] preview_port, 8000)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    # Drop the software limits: full 0-180 servo travel on both axes.
    cfg.pan = dataclasses.replace(cfg.pan, min_angle=FULL_MIN, max_angle=FULL_MAX)
    cfg.tilt = dataclasses.replace(cfg.tilt, min_angle=FULL_MIN, max_angle=FULL_MAX)

    with PanTilt(cfg) as pt:
        pt.jump(
            pan=args.start_pan if args.start_pan is not None else cfg.pan.center,
            tilt=args.start_tilt if args.start_tilt is not None else cfg.tilt.center,
        )
        explorer = Explorer(pt, args.step, Path(args.marks) if args.marks else None)
        explorer._track()

        stream = None
        if args.preview:
            try:
                from pantilt.stream import CameraStream

                stream = CameraStream(cfg.camera, status=explorer.status, port=args.port)
                explorer.preview_url = stream.start()
            except ImportError:
                print("picamera2 not found. Install it with `sudo apt install python3-picamera2`\n"
                      "and make sure the venv was created with --system-site-packages.")
                return 1
            except (RuntimeError, OSError, IndexError) as e:  # IndexError = no camera found
                if stream:
                    stream.stop()
                print(f"Couldn't start the camera preview: {e}\n"
                      "Is the camera connected, and is another program (rpicam-hello, a scan) using it?")
                return 1

        try:
            curses.wrapper(run, explorer)
        except KeyboardInterrupt:
            pass
        finally:
            if stream:
                stream.stop()
        print(explorer.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
