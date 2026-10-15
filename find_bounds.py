#!/usr/bin/env python3
"""find_bounds.py - drive the pan-tilt freely with the arrow keys and watch the live angles.

There are NO config limits here. By default each servo can go from -30 to 210
degrees, i.e. past the servo's rated 0-180. The degrees are extended past
0-180 using the same scale (500 us = 0 deg, 2500 us = 180 deg by default),
so the pulse width (us) shown next to each angle is the true command being
sent.

Past 0/180 most servos hit their internal end stop. The angle on screen keeps
changing, but the horn stops moving. Where it stops is that servo's real
physical limit. If a servo buzzes, strains or gets warm at an end, back off
right away or press r. A servo stalled against its stop can strip its gears.

Channels, direction (invert) and pulse widths come from config.toml. Only the
angle limits are ignored.

Run from the repo folder:
    python find_bounds.py
    python find_bounds.py --min-angle -45 --max-angle 225   # go even further
    python find_bounds.py --step 5 --start-pan 90 --start-tilt 90

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
from datetime import datetime
from pathlib import Path

from pantilt import load_config
from pantilt.config import ServoConfig
from pantilt.controller import PanTilt

RATED_MIN, RATED_MAX = 0.0, 180.0
PULSE_FLOOR_US, PULSE_CEIL_US = 100, 3000  # hard safety cap on the raw pulse
STEPS = [0.5, 1, 2, 5, 10, 20]


def pulse_for(cfg: ServoConfig, angle: float) -> float:
    """Same scale as the library, extended past 0-180, capped to a sane pulse range."""
    a = 180 - angle if cfg.invert else angle
    span = cfg.max_pulse_us - cfg.min_pulse_us
    pulse = cfg.min_pulse_us + span * a / 180
    return max(PULSE_FLOOR_US, min(PULSE_CEIL_US, pulse))


class Explorer:
    def __init__(self, pt: PanTilt, step: float, lo: float, hi: float,
                 start: tuple[float, float], marks_file: Path | None):
        self.pt = pt
        self.lo, self.hi = lo, hi
        self.step_i = min(range(len(STEPS)), key=lambda i: abs(STEPS[i] - step))
        self.marks: list[tuple[float, float]] = []
        self.marks_file = marks_file
        self.angle = {"pan": start[0], "tilt": start[1]}
        self.seen = {k: [v, v] for k, v in self.angle.items()}
        self.released = False
        self.msg = ""
        self._send()

    @property
    def step(self) -> float:
        return STEPS[self.step_i]

    def cfg(self, axis: str) -> ServoConfig:
        return getattr(self.pt.config, axis)

    def pulse(self, axis: str) -> float:
        return pulse_for(self.cfg(axis), self.angle[axis])

    def _send(self):
        for axis in ("pan", "tilt"):
            self.pt.driver.set_pulse_us(self.cfg(axis).channel, self.pulse(axis))
        self.released = False

    def move(self, axis: str, delta: float):
        target = self.angle[axis] + delta
        if target < self.lo or target > self.hi:
            target = max(self.lo, min(self.hi, target))
            self.msg = f"{axis} at the {self.lo:g}..{self.hi:g} deg limit (change with --min-angle/--max-angle)"
        self.angle[axis] = target
        lo, hi = self.seen[axis]
        self.seen[axis] = [min(lo, target), max(hi, target)]
        self._send()

    def mark(self):
        p, t = self.angle["pan"], self.angle["tilt"]
        self.marks.append((p, t))
        self.msg = f"mark {len(self.marks)} saved: pan={p:.1f} tilt={t:.1f}"
        if self.marks_file:
            new = not self.marks_file.exists()
            with self.marks_file.open("a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "pan_deg", "tilt_deg", "pan_pulse_us", "tilt_pulse_us"])
                w.writerow([datetime.now().isoformat(timespec="seconds"), f"{p:.1f}", f"{t:.1f}",
                            f"{self.pulse('pan'):.0f}", f"{self.pulse('tilt'):.0f}"])

    def handle_key(self, k: int) -> bool:
        """Apply one key press. Returns False when it's time to quit."""
        self.msg = ""
        s = self.step
        if k == curses.KEY_UP:
            self.move("tilt", +s)
        elif k == curses.KEY_DOWN:
            self.move("tilt", -s)
        elif k == curses.KEY_RIGHT:
            self.move("pan", +s)
        elif k == curses.KEY_LEFT:
            self.move("pan", -s)
        elif k in (ord("+"), ord("=")):
            self.step_i = min(self.step_i + 1, len(STEPS) - 1)
        elif k in (ord("-"), ord("_")):
            self.step_i = max(self.step_i - 1, 0)
        elif k in (ord("m"), ord("M")):
            self.mark()
        elif k in (ord("c"), ord("C")):
            self.angle = {"pan": self.cfg("pan").center, "tilt": self.cfg("tilt").center}
            self._send()
        elif k in (ord("r"), ord("R")):
            self.pt.release()
            self.released = True
            self.msg = "released - servos limp. Press an arrow key to re-engage."
        elif k in (ord("q"), ord("Q"), 27):
            return False
        return True

    def _row(self, axis: str) -> str:
        a = self.angle[axis]
        lo, hi = self.seen[axis]
        flag = "  << PAST RATED 0-180" if not RATED_MIN <= a <= RATED_MAX else ""
        return (f"  {axis.upper():<4}  {a:7.1f} deg   pulse {self.pulse(axis):5.0f} us"
                f"   visited {lo:6.1f} .. {hi:6.1f}{flag}")

    def lines(self) -> list[str]:
        state = "RELEASED (limp)" if self.released else "holding"
        out = [
            f"Pan-tilt bounds finder  (range {self.lo:g}..{self.hi:g} deg - watch the bracket!)",
            "",
            self._row("pan"),
            self._row("tilt"),
            "",
            f"  step {self.step} deg    servos {state}    marks {len(self.marks)}",
            "",
            "  arrows move   +/- step   m mark   c center   r release   q quit",
            "  Past 0/180: if the angle changes but the horn doesn't move, you're at the end stop.",
            "",
        ]
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
                    "If your marks are the safe edges, use these in config.toml:",
                    f"  [pan]   min_angle = {max(min(ps), 0):.0f}   max_angle = {min(max(ps), 180):.0f}",
                    f"  [tilt]  min_angle = {max(min(ts), 0):.0f}   max_angle = {min(max(ts), 180):.0f}",
                ]
                if min(ps + ts) < RATED_MIN or max(ps + ts) > RATED_MAX:
                    out.append("  (config limits must stay within 0-180, so marks past that were clipped.\n"
                               "   To reach them in normal use, widen min_pulse_us/max_pulse_us in config.toml.)")
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
    ap = argparse.ArgumentParser(description="Drive the pan-tilt past its limits and read live angles.")
    ap.add_argument("--config", help="config.toml to read channels/invert/pulses from")
    ap.add_argument("--min-angle", type=float, default=-30, help="lowest angle allowed (default -30)")
    ap.add_argument("--max-angle", type=float, default=210, help="highest angle allowed (default 210)")
    ap.add_argument("--step", type=float, default=2, help="starting step in degrees (default 2)")
    ap.add_argument("--start-pan", type=float, help="starting pan angle (default: config center)")
    ap.add_argument("--start-tilt", type=float, help="starting tilt angle (default: config center)")
    ap.add_argument("--marks", default="marks.csv", help="file to save marks to ('' to disable)")
    args = ap.parse_args(argv)
    if args.min_angle >= args.max_angle:
        ap.error("--min-angle must be less than --max-angle")

    cfg = load_config(args.config)
    start = (
        args.start_pan if args.start_pan is not None else cfg.pan.center,
        args.start_tilt if args.start_tilt is not None else cfg.tilt.center,
    )

    with PanTilt(cfg) as pt:
        explorer = Explorer(pt, args.step, args.min_angle, args.max_angle, start,
                            Path(args.marks) if args.marks else None)
        try:
            curses.wrapper(run, explorer)
        except KeyboardInterrupt:
            pass
        print(explorer.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
