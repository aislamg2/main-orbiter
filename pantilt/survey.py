"""Scan the board in board coordinates, classify each tile, save the hits.

This is `scan` and `live` joined up: step to a pose, let the mount settle,
capture a still, run it through the Hailo classifier, and write the frame out
when the crater probability clears a threshold. Every tile gets a CSV row with
all eight class probabilities, so a threshold can be re-chosen afterwards
without re-running the scan.

    pantilt measure-k --separation 1000   # fills in [board] mm_per_px
    pantilt survey --dry-run              # print the plan, move nothing
    pantilt survey
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .classify import CLASS_NAMES, HailoClassifier, preprocess
from .config import BoardConfig, Config
from .controller import PanTilt
from .geometry import describe, scan_order, tile_mm

CRATER_INDEX = CLASS_NAMES.index("crater")


# -- camera ------------------------------------------------------------------
class StillCamera:
    """Still capture with exposure frozen, so two tiles of the same terrain match.

    Auto-exposure re-adapting between poses would hand the classifier brightness
    changes that aren't in the scene. Let it settle once, read what it chose,
    then pin it for the rest of the run.
    """

    def __init__(self, cfg: Config, settle_s: float = 2.5):
        from picamera2 import Picamera2

        from .stream import camera_transform

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_still_configuration(
            transform=camera_transform(cfg.camera)))
        self.picam2.start()
        time.sleep(settle_s)

        md = self.picam2.capture_metadata()
        self.exposure_us = int(md["ExposureTime"])
        self.gain = float(md["AnalogueGain"])
        self.colour_gains = tuple(float(x) for x in md["ColourGains"])
        self.picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": self.exposure_us,
            "AnalogueGain": self.gain,
            "ColourGains": self.colour_gains,
            "NoiseReductionMode": 0,
            "Sharpness": 1.0,
            "Contrast": 1.0,
        })
        time.sleep(1.5)

    def grab(self) -> np.ndarray:
        return self.picam2.capture_array("main")

    @property
    def warning(self) -> str | None:
        if self.gain > 3.0:
            return (f"analogue gain {self.gain:.1f} - the board is underlit. Every tile "
                    "will carry sensor noise the model was never trained on. Add light "
                    "and re-run.")
        if self.exposure_us > 20_000:
            return (f"exposure {self.exposure_us / 1000:.0f} ms - long enough that leftover "
                    "mount vibration will smear tiles. Add light.")
        return None

    def settings(self) -> dict:
        return {"exposure_us": self.exposure_us, "analogue_gain": self.gain,
                "colour_gains": list(self.colour_gains)}

    def close(self):
        self.picam2.stop()
        self.picam2.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# -- measure-k ----------------------------------------------------------------
def cmd_measure_k(args, cfg: Config) -> int:
    """Measure mm_per_px, the number the whole scan geometry rests on.

    Tape two marks a known distance apart on the board, horizontally, both
    inside the frame. They don't have to be on the printed map - masking tape on
    a bare board works fine, which means this can be done before the print
    arrives.
    """
    import cv2

    with StillCamera(cfg) as cam:
        if cam.warning:
            print(f"warning: {cam.warning}\n", file=sys.stderr)
        frame = cam.grab()

    out = Path(args.out)
    cv2.imwrite(str(out), frame[:, :, :3][:, :, ::-1])  # RGB -> BGR for imwrite
    h, w = frame.shape[:2]
    print(f"saved {out}  ({w} x {h} px)")
    print("\nOpen it, find the two marks, read off their x pixel values.")
    print("Tip: `python3 -m http.server 8000` on the Pi, then browse it.\n")

    try:
        x1 = float(input("x pixel of LEFT mark:  "))
        x2 = float(input("x pixel of RIGHT mark: "))
    except (EOFError, ValueError):
        print("cancelled")
        return 1

    px = abs(x2 - x1)
    if px < 10:
        print("those are too close to be two different marks", file=sys.stderr)
        return 1

    k = args.separation / px
    print(f"\n{px:.0f} px across {args.separation:.0f} mm")
    print(f"full frame covers {w * k:.0f} x {h * k:.0f} mm of board")
    print(f"print the HiRISE source at {25.4 / k:.0f} DPI for 1:1 pixel sampling")
    print(f"\nPut this in config.toml under [board]:\n\n    mm_per_px = {k:.5f}\n")
    return 0


# -- survey -------------------------------------------------------------------
def _probe_frame_shape(cfg: Config) -> tuple[int, int]:
    """Frame height and width, without holding the camera open for the whole plan."""
    with StillCamera(cfg, settle_s=0.5) as cam:
        return cam.grab().shape[:2]


def cmd_survey(args, cfg: Config) -> int:
    board: BoardConfig = cfg.board
    if not board.measured:
        print("[board] mm_per_px is 0 - run `pantilt measure-k` first.", file=sys.stderr)
        return 1

    hef = Path(args.hef or cfg.model.hef_path)
    if not hef.exists():
        print(f"model not found: {hef}", file=sys.stderr)
        return 1

    crop = args.crop if args.crop is not None else cfg.model.crop
    threshold = args.threshold

    fh, fw = _probe_frame_shape(cfg)
    tile = tile_mm(board, fh, fw, crop)
    tiles = list(scan_order(board, tile, serpentine=args.serpentine))

    print(describe(board, tile))
    print(f"frame        {fw} x {fh} px, classifying the centre {crop:.0%} square")
    print(f"trigger      crater probability >= {threshold:.2f}\n")

    # Refuse to start a plan the bracket can't physically reach.
    out_of_range = [t for t in tiles
                    if not cfg.pan.min_angle <= t.pan_deg <= cfg.pan.max_angle
                    or not cfg.tilt.min_angle <= t.tilt_deg <= cfg.tilt.max_angle]
    if out_of_range:
        t = out_of_range[0]
        print(f"{len(out_of_range)} of {len(tiles)} tiles fall outside the servo limits, "
              f"e.g. tile {t.index}: pan {t.pan_deg:.1f}, tilt {t.tilt_deg:.1f}", file=sys.stderr)
        print("Move the camera back, shrink [board] width/height, or widen the limits.",
              file=sys.stderr)
        return 1

    if args.dry_run:
        for t in tiles[:8]:
            print(f"  {t.index:>3}  r{t.row} c{t.col}  ({t.x_mm:>8.1f}, {t.y_mm:>8.1f}) mm"
                  f"  ->  pan {t.pan_deg:>6.2f}  tilt {t.tilt_deg:>6.2f}")
        if len(tiles) > 8:
            print(f"  ... {len(tiles) - 8} more")
        return 0

    run = Path(args.out or f"data/runs/{datetime.now():%Y%m%d-%H%M%S}")
    (run / "hits").mkdir(parents=True, exist_ok=True)
    if args.save_all:
        (run / "tiles").mkdir(parents=True, exist_ok=True)

    import cv2

    hits = 0
    t_start = time.time()
    with StillCamera(cfg) as cam, HailoClassifier(str(hef)) as clf, PanTilt(cfg) as pt:
        if cam.warning:
            print(f"warning: {cam.warning}\n", file=sys.stderr)

        (run / "manifest.json").write_text(json.dumps({
            "started": datetime.now().isoformat(timespec="seconds"),
            "board": asdict(board),
            "camera": cam.settings(),
            "frame": [fw, fh],
            "crop": crop,
            "tile_mm": tile,
            "threshold": threshold,
            "repeat": args.repeat,
            "settle_s": cfg.motion.settle_s,
            "serpentine": args.serpentine,
            "hef": str(hef),
            "classes": CLASS_NAMES,
            "tiles": len(tiles),
        }, indent=2))

        with (run / "tiles.csv").open("w", newline="") as fh_csv:
            w = csv.writer(fh_csv)
            w.writerow(["index", "row", "col", "x_mm", "y_mm", "pan_deg", "tilt_deg",
                        "t_unix", "crater_p", "top_label", "top_p", "detected", "file"]
                       + [f"p_{c.replace(' ', '_')}" for c in CLASS_NAMES])

            for t in tiles:
                pt.move_to(pan=t.pan_deg, tilt=t.tilt_deg)
                time.sleep(cfg.motion.settle_s)

                # Averaging repeats at one pose cuts sensor noise without mixing
                # terrain from neighbouring tiles, which cross-frame smoothing would.
                frame = None
                acc = np.zeros(len(CLASS_NAMES), np.float32)
                for _ in range(args.repeat):
                    frame = cam.grab()
                    acc += clf.classify(preprocess(frame, crop=crop))
                probs = acc / args.repeat

                crater_p = float(probs[CRATER_INDEX])
                top = int(np.argmax(probs))
                detected = crater_p >= threshold
                name = ""
                if detected or args.save_all:
                    sub, tag = ("hits", "crater") if detected else ("tiles", "tile")
                    name = f"t{t.index:04d}_r{t.row}c{t.col}_{tag}_{crater_p:.2f}.png"
                    cv2.imwrite(str(run / sub / name), frame[:, :, :3][:, :, ::-1])
                hits += detected

                w.writerow([t.index, t.row, t.col, f"{t.x_mm:.2f}", f"{t.y_mm:.2f}",
                            f"{t.pan_deg:.3f}", f"{t.tilt_deg:.3f}", f"{time.time():.3f}",
                            f"{crater_p:.4f}", CLASS_NAMES[top], f"{probs[top]:.4f}",
                            int(detected), name]
                           + [f"{p:.4f}" for p in probs])
                fh_csv.flush()

                flag = "  <-- CRATER" if detected else ""
                print(f"  {t.index + 1:>3}/{len(tiles)}  pan {t.pan_deg:>6.2f} "
                      f"tilt {t.tilt_deg:>6.2f}  crater {crater_p:>5.2f}  "
                      f"top {CLASS_NAMES[top]:<14}{flag}")

        pt.center()

    print(f"\n{hits} crater{'' if hits == 1 else 's'} in {len(tiles)} tiles, "
          f"{time.time() - t_start:.0f}s -> {run}")
    if hits:
        print(f"images in {run / 'hits'}")
    print("All eight probabilities are in tiles.csv, so you can re-threshold "
          "offline without re-scanning.")
    return 0
