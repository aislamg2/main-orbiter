# Command reference

Every command, in one place. Run them on the Pi from the project folder with
the venv active:

```bash
cd ~/main-orbiter
source .venv/bin/activate
```

`pantilt <command>` works after `pip install -e .`. Otherwise use
`python3 -m pantilt <command>`, never `python3 pantilt ...`. Add `--help` to
any command to see all of its options.

When a command prints `live view: http://<pi-ip>:8000/`, open that address in a
browser on the same network to watch the camera, the red crop box and the live
scores.

## Crater hunting

| What | Command |
|---|---|
| Full demo (teaches and calibrates only if needed, then scans) | `./scripts/demo.sh --threshold 0.5` |
| Full demo, re-teaching the board corners first | `./scripts/demo.sh --teach --threshold 0.5` |
| **Scan the board, go back to likely craters, photograph them** | `pantilt board-scan --threshold 0.5` |
| Same, with exposure frozen for the whole run | `pantilt board-scan --threshold 0.5 --lock-exposure` |
| Find from one wide view instead of walking the board (faster) | `pantilt board-scan --wide --threshold 0.5` |
| Old scan: one look per square, no going back | `pantilt board-scan --grid --threshold 0.5` |
| Keep scanning until Ctrl+C | `pantilt board-scan --passes 0` |
| Print the plan, move nothing | `pantilt board-scan --dry-run` |

Other `board-scan` options:

| Option | Meaning |
|---|---|
| `--crop 0.2` | Size of the square the model judges (fraction of the frame height) |
| `--threshold 0.6` | Crater score needed for a photo, after going back |
| `--speed 20` | Degrees per second between stops (slower shakes less) |
| `--max-photos 20` | Photo limit per run |
| `--cooldown-tiles 0.5` / `--cooldown-s 3` | No photo within half a square / 3 s of the last |
| `--out data/runs/myrun` | Where to save the run |
| `--no-preview` / `--port 8001` | No browser live view / a different port |
| `--hef models/other.hef` | Use a different model |

Settings without a flag live under `[sweep]` in `config.toml`:

| Setting | Meaning |
|---|---|
| `look_threshold` | Crater score during the scan that makes a spot worth going back to (0.15) |
| `watch_min_s` | Minimum time classifying at each stop and revisit (0.8 s) |
| `watch_max_s` / `revisit_watch_max_s` | How long it may keep watching while the score is still climbing (3 s / 4 s) |
| `find_frames` / `confirm_frames` | Frames averaged for a score at a scan stop / revisit (5 / 10) |
| `settle_s` / `settle_max_s` | Wait after each move / longest wait for the shaking to stop |
| `refine_steps` | Times a revisit may re-aim towards a better square |
| `overlap` / `max_gap` | How the scan squares tile the board |
| `edge_margin` | How far a square may hang off the taught outline (0.4), so craters at the edge can be centred |

## Setting up the board

| What | When | Command |
|---|---|---|
| Teach the 4 board corners | The board or camera moved | `pantilt teach` (add `--gui` for a window) |
| Measure the servo step and directions | New camera, lens or servos | `pantilt calibrate-step` |
| Walk every scan stop, no classifying | Check the coverage | `pantilt board-scan --trace` |
| Visit the 4 outermost stops | Check the box reaches the board edges | `pantilt board-scan --trace-edges` |

In `teach`: the arrow keys aim, `+`/`-` change the step size, `m` (or space)
marks a corner, `u` undoes, `g` re-visits the marks, `s` saves, `q` quits. Mark
the corners in order going around the edge of the printed map, on the **outer**
corners of the outermost sheets. Craters close to the outline are only checked
properly if the outline reaches them.

## Looking around by hand

| What | Command |
|---|---|
| Live classification, aim with the arrow keys | `pantilt live` |
| Live classification in a window: camera, red box, class and confidence, buttons | `pantilt live --gui` (needs `sudo apt install python3-tk` and a screen) |
| Live, at a different crop | `pantilt live --crop 0.3` |
| Live, logging predictions to a CSV | `pantilt live --log data/live.csv` |
| Live, no keyboard, print one prediction per second | `pantilt live --no-keys` |
| Camera only, no model | `python3 scripts/preview.py` |
| Free movement past the angle limits, showing angles | `python3 find_bounds.py --preview` |

In `live`: the arrow keys aim, `+`/`-` change the step, `c` centres, `r`
releases the servos, `q` quits. With `--gui` the same keys work in the window,
and each one also has a button. `--log`, `--crop` and `--step` work as usual,
and the browser live view runs alongside unless you add `--no-preview`.

## Servos and hardware

| What | Command |
|---|---|
| Check the I2C bus, servo board and Hailo | `pantilt check` |
| Centre both servos | `pantilt center` |
| Go to an angle (pan, tilt) | `pantilt goto 100 90` |
| Drive with the arrow keys (no camera) | `pantilt keys` |
| Drive with the arrow keys in a window with the live camera | `pantilt keys --gui` (needs `sudo apt install python3-tk`) |
| Test each servo's full range | `pantilt sweep` (or `--axis pan` / `--axis tilt`) |
| Release the servos (go limp) | `pantilt off` |

## Results

Each `board-scan` run saves to `data/runs/board-<date>-<time>/`:

| File | What's in it |
|---|---|
| `photos/` | Each crater photo (`.jpg`) and the exact square the model judged (`_crop.png`) |
| `overview_p1_0.jpg` | The whole board: outline (blue), scan stops (magenta), photos (green), near misses (orange), skipped duplicates (grey, labelled at the bottom) |
| `candidates.csv` | Every revisit: scan score, revisit score, seconds watched, result |
| `windows.csv` | Every square scored during the scan |
| `run.json` | The settings used |

| What | Command |
|---|---|
| View a run's images in a browser | `cd data/runs/<run> && python3 -m http.server 8080`, then open `http://<pi-ip>:8080/` |
| A run's revisit scores | `cut -d, -f2,5,6,8,9 data/runs/<run>/candidates.csv` |
| List runs and photo counts (deletes nothing) | `pantilt clean` |
| Delete photos from all but the newest 3 runs | `pantilt clean --keep 3` |
| Delete photos from every run | `pantilt clean --all` |
| Delete one run completely | `pantilt clean --run board-20261022-163257 --everything` |

`clean` always lists what it will delete and asks first (`--yes` skips the
question). Deleted files can't be recovered.

## Getting the latest code

```bash
git stash && git pull && git stash pop
```

The stash keeps your local `config.toml` edits (such as `crop = 0.2`) across
the pull. Your taught corners and calibration (`data/board_*.json`) are never
touched by git.

## Older tools

| What | Command |
|---|---|
| Fixed-angle grid of photos | `pantilt scan --pan 90 120 --tilt 80 100 --cols 5 --rows 3 --capture scans/run1` |
| Measure mm per pixel from two marks | `pantilt measure-k --separation 1000` |
| Survey using measured board geometry | `pantilt survey --dry-run` |
