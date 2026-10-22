# orbiter-pantilt

Pan-tilt camera control for the crater-detection orbiter. The camera sits on a
two-servo pan-tilt bracket driven by a PCA9685 servo board over I2C, on a
Raspberry Pi 5 with the Raspberry Pi AI HAT+ (Hailo-8) stacked on top.

The code talks to the PCA9685 directly through `/dev/i2c-1` using `smbus2`, so
it needs no GPIO library (those are flaky on the Pi 5) and never touches the pins
the AI HAT+ uses.

## Hardware

| Part | Notes |
|---|---|
| Raspberry Pi 5 | Raspberry Pi OS Bookworm or newer |
| Raspberry Pi AI HAT+ (26 TOPS) | PCIe ribbon + stacking header, with a GPIO extender through it |
| PCA9685 servo driver | I2C address `0x40` |
| Pan-tilt bracket + 2 servos | ArduCAM pan-tilt kit |
| External 5–6 V supply | Powers the servos via the driver's **V+** terminal |

### Wiring

| PCA9685 | Raspberry Pi header |
|---|---|
| VCC | Pin 1 (3.3 V) – logic only |
| SDA | Pin 3 (GPIO 2, I2C1 SDA) |
| SCL | Pin 5 (GPIO 3, I2C1 SCL) |
| GND | Pin 6 (any GND) |

| Servo | PCA9685 channel |
|---|---|
| Tilt (up/down) | 0 |
| Pan (left/right) | 1 |

**Do not use header pins 27/28** (GPIO 0/1). They are the HAT+ ID EEPROM bus that
the AI HAT+ uses. **Do not power the servos from the Pi's 5 V pins.** The Hailo
already draws from that rail, and servo current spikes cause brown-outs. Feed
the servos from the external supply into V+, with grounds shared.

## Setup (on the Pi)

```bash
# 1. Enable I2C and give your user access to it
sudo raspi-config nonint do_i2c 0
sudo usermod -aG i2c $USER          # log out and back in after this

# 2. Clone and install
git clone https://github.com/<your-username>/orbiter-pantilt.git
cd orbiter-pantilt
python3 -m venv --system-site-packages .venv   # system packages = picamera2 + Hailo SDK
source .venv/bin/activate
pip install -e .

# 3. Check the hardware
pantilt check
```

`--system-site-packages` matters: `picamera2` and the Hailo Python SDK
(`hailo-all`) are installed through apt, and a plain venv can't see them.

## Usage

```bash
pantilt check                 # I2C bus, PCA9685 at 0x40, Hailo on PCIe
pantilt center                # both servos to 90°
pantilt goto 120 60           # pan=120°, tilt=60°
pantilt sweep                 # each axis min -> max -> center (range/direction test)
pantilt sweep --axis tilt
pantilt keys                  # arrow keys to aim, +/- step size, c center, r release, q quit
pantilt off                   # release both servos (go limp, stop drawing current)

# Raster-scan the map: 5 x 3 grid, saving a photo + poses.csv at each stop
pantilt scan --pan 40 140 --tilt 50 110 --cols 5 --rows 3 --capture scans/run1
```

Without `pip install -e .`, run any command as `python -m pantilt <command>` from
the repo folder.

The servos keep holding their position after a command exits. Run
`pantilt off` to release them.

### Finding the camera's bounds

```bash
python find_bounds.py
```

Drive the camera with the arrow keys over the full 0–180° range of each servo.
The angle limits in `config.toml` are ignored here. The screen shows the live
pan/tilt angles, the pulse width, and the range you've visited. Press **m** to
mark a position (saved to `marks.csv`). When you quit with **q**, it prints
your marks plus suggested `config.toml` limits and a matching `pantilt scan`
command. Other keys: **+/-** step size, **c** center, **r** release.

With no limits, the tilt bracket can hit its frame. If a servo buzzes or
strains, back off or press **r**.

**Live camera view:** add `--preview`:

```bash
python find_bounds.py --preview
```

The terminal shows an address like `http://192.168.1.42:8000/`. Open it in a
browser on your Mac (same network) or on the Pi. You get the live camera with a
center crosshair and the current pan/tilt angles drawn on top. Only one program
can use the camera at a time, so close `rpicam-hello` or any running scan first.

### Camera orientation

The camera is mounted upside down, so `config.toml` has:

```toml
[camera]
rotate_180 = true
```

This flips the live preview and the photos from `pantilt scan --capture`. For
the `rpicam-*` tools, add `--rotation 180`, e.g. `rpicam-hello -t 0 --rotation 180`.

## Using it from Python (e.g. the crater-detection pipeline)

```python
from pantilt import PanTilt, load_config, raster

with PanTilt(load_config("config.toml")) as cam_mount:
    cam_mount.center()
    for pan, tilt in raster((40, 140), (50, 110), cols=5, rows=3):
        cam_mount.move_to(pan=pan, tilt=tilt, settle=True)
        # grab a frame here and run the Hailo model on it
    cam_mount.center()
```

`move_to()` moves smoothly at `speed_dps`. `jump()` goes there instantly.
`settle=True` waits `settle_s` so the frame isn't blurred by vibration.

## Real-time landmark classification (Hailo-8)

`pantilt live` runs the Mars landmark classifier from
[mars-landmark-classification](https://github.com/reemshehayib/mars-landmark-classification)
(`hailo-inference` branch) on every camera frame, on the AI HAT+, while you aim
the camera.

```bash
# one-time: the Hailo Python bindings, and the compiled model
sudo apt install python3-hailort
mkdir -p models
cp ~/hailo-inference/mars-landmark-classification/mars_model_float32.hef models/mars_landmark.hef

pantilt live                          # arrow keys aim, live label in the terminal + browser
pantilt live --log runs/live.csv      # also save predictions with pan/tilt (one row per second)
pantilt live --crop 0.5               # classify only the middle of the frame (around the crosshair)
pantilt live --no-keys --no-preview   # headless: print one prediction per second

# raster-scan the map and classify each pose
pantilt live --scan --pan 40 140 --tilt 50 110 --cols 5 --rows 3 --log runs/scan1.csv
```

The browser live view (`http://<pi-ip>:8000/`) shows the predicted class,
confidence, the next two candidates, FPS, and the Hailo inference time.

How each frame is prepared, to match the training tiles: the largest centered
square (or `crop` of it) is cut out, converted to grayscale, resized to
224 x 224 and fed to the Hailo as uint8. The normalization is built into the
HEF. `smooth_frames` averages the last few predictions so the label doesn't
flicker. Settings are in the `[model]` section of `config.toml`.

Use the HEF rebuilt with the real-image calibration set (`make_calib_set.py` +
`mars_landmark.alls`). The older HEF expects a different input range and will
give wrong labels.

## Scanning the board for craters (`teach` + `board-scan`)

This does what `pantilt live --crop 0.2 --log ...` does, but the camera walks
itself over the board. It stops at each grid point, waits until the picture is
sharp, and saves a photo when it sees a crater.

**1. Teach the board's corners (once, and again if the camera or board moves).**

```bash
pantilt teach --crop 0.2
```

Open the live view, then use the arrow keys to put the green crosshair on each
corner of the printed map. Go around the edge in order: any start corner, either
direction. Press **m** at each corner, **g** to re-visit them, **u** to undo,
and **s** to save to `data/board_corners.json`. The scan never leaves this shape,
so nothing off the board gets photographed.

**2. Measure the step (once, and again if you change the camera, lens or servos).**

```bash
pantilt calibrate-step --crop 0.2
```

The camera moves 5° at a time on each axis and measures how far the image
shifts, so the grid uses the real size of one crop square in servo degrees.
Until you run this, the step is estimated from the lens field of view, and
there may be thin gaps or overlaps between squares.

**3. Check the plan, then check it on the hardware.**

```bash
pantilt board-scan --crop 0.2 --dry-run       # grid size and angles, moves nothing
pantilt board-scan --crop 0.2 --trace         # walks every grid point, no classifying
pantilt board-scan --crop 0.2 --trace-edges   # just the 4 outermost points
```

During `--trace`, each red box should start where the previous one ended. The
grid uses the fewest squares that cover the board. They butt together with no
overlap, and any leftover space is spread evenly as a thin sliver between
squares, so there's never a gap and never a square off the board. When the
leftover is tiny (`max_gap`), it drops that row or column instead and leaves a
thin unscanned strip at the edges. During
`--trace-edges`, the box should reach the board edge but stay on it.

**4. Scan.**

```bash
pantilt board-scan --crop 0.2                       # one pass, threshold from config.toml
pantilt board-scan --crop 0.2 --threshold 0.75      # stricter
pantilt board-scan --crop 0.2 --passes 0            # keep going until Ctrl+C
```

Each run writes to `data/runs/board-<timestamp>/`:

- `photos/`: the full frame (`.jpg`) plus the exact crop the model judged
  (`_crop.png`), named by pose and confidence
- `poses.csv`: every pose with all 8 probabilities and what happened there
  (`saved`, `below`, `near`, `wait`, `cap`), so you can pick a better threshold
  afterwards without re-scanning
- `run.json`: the settings used for the run

How it avoids blur and duplicate photos (all set in `[sweep]` in `config.toml`):

| Setting | What it does |
|---|---|
| `speed_dps`, `settle_s` | Moves at 30°/s with eased starts and stops, then always waits 0.4 s |
| `settle_max_s`, `still_diff` | Then keeps waiting until two frames in a row look the same (up to 2 s). `still` = 0 in the CSV means it gave up |
| `threshold` | Crater probability needed for a photo (`--threshold`) |
| `cooldown_tiles` | No new photo within half a square of one already saved, which stops repeats of the same square on later `--passes` |
| `cooldown_s` | Minimum seconds between photos (0 = off, since squares no longer overlap) |
| `max_photos` | Hard cap per run |
| `overlap` | Minimum share between neighbouring squares (0 = edge to edge; try 0.2 if craters on a boundary get missed) |
| `max_gap` | If one row or column fewer would leave at most this much of a square unscanned (split between both edges), use one fewer. Stops a board 3.15 squares tall getting 4 heavily overlapping rows. 0 = always cover every millimetre |
| `hfov_deg` | Lens field of view, only used until `calibrate-step` has run |

If the CSV shows a lot of `still = 0`, slow down (`speed_dps`) or raise
`settle_max_s`. If a still bracket never counts as still (each pose takes the
full `settle_max_s`), raise `still_diff`.

Photos are saved at the preview resolution (`preview_width` x `preview_height`).
At `--crop 0.2` the crop is only 144 px square, so raise the preview size if you
want sharper saved crops.

## Configuration

All settings live in [`config.toml`](config.toml): channels, angle limits, pulse
widths, direction, and speed. Common tweaks:

- **Arrow keys move the wrong way:** set `invert = true` on that axis.
- **Servo buzzes or strains at an end:** pull in `min_angle`/`max_angle`, or
  narrow `min_pulse_us`/`max_pulse_us` (try 600/2400).
- **Servos are swapped:** swap the `channel` numbers, or swap the plugs.
- **Driver at a different address:** change `address` (check with `i2cdetect -y 1`).

The tilt limits (15°–145°) match the ArduCAM bracket's mechanical range. Don't
widen them without checking that nothing binds.

## Tests

The unit tests use a fake I2C bus, so they run on any machine:

```bash
pip install -e ".[dev]"
pytest
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/dev/i2c-1 missing` | `sudo raspi-config nonint do_i2c 0`, then reboot |
| `permission denied on the I2C bus` | `sudo usermod -aG i2c $USER`, then log out/in |
| `no PCA9685 at 0x40` | Check SDA (pin 3) / SCL (pin 5) / VCC / GND; run `i2cdetect -y 1` |
| Command runs, servos don't move | No power on V+, or servo plug reversed (brown/black = GND) |
| Pi reboots or Hailo disappears when servos move | Servo power is reaching the Pi's 5 V rail; use only the external V+ supply |
| Preview page won't load from the Mac | Use the Pi's IP (`hostname -I`), same network, port 8000; campus networks can block device-to-device traffic, so try a phone hotspot or open it in the Pi's own browser |
| `Couldn't start the camera preview` | Camera not detected (`rpicam-hello --list-cameras`) or another program is using it |

## Project layout

```
pantilt/
  pca9685.py     PCA9685 register-level driver (smbus2)
  controller.py  Servo + PanTilt: limits, smooth moves, center, release
  config.py      config.toml loading and defaults
  scan.py        raster (serpentine) scan pattern
  stream.py      live camera preview as a web page (MJPEG)
  classify.py    Hailo-8 landmark classifier + live classification loop
  board_scan.py  `teach` corners, `calibrate-step`, `board-scan` for craters
  cli.py         `pantilt` command line
config.toml      hardware and model settings
models/          compiled Hailo model (.hef) for `pantilt live`
find_bounds.py   free arrow-key control with live angles (no limits)
tests/           hardware-free unit tests
```

## Credits

Channel layout, angle limits, and pulse range follow ArduCAM's pan-tilt demo
([ArduCAM/PCA9685](https://github.com/ArduCAM/PCA9685)), which is still a handy
reference: its C example (`example/rpi`) also works on this setup.
