"""Gui for aiming the camera and teaching the board corners.

`pantilt keys --gui` and `pantilt teach --gui` opens a gui instead of the curses
screen, and `pantilt live --gui` adds the live classification to it. It has to
run where the camera is (the pi or through ssh with X forwarding).
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from pathlib import Path

from .board_scan import STEPS, Corner, save_corners
from .config import Config
from .controller import PanTilt

class AimSession:
    """ Move, change step, mark / undo / revisit / save corners."""

    def __init__(self, pt, teach: bool, corners_file: str | Path, step: float = 2):
        self.pt = pt
        self.teach = teach
        self.corners_file = Path(corners_file)
        self.step_i = min(range(len(STEPS)), key=lambda i: abs(STEPS[i] - step))
        self.corners: list[Corner] = []
        self.msg = ""
        self.saved = False
        self.busy = False  # True while revisiting the marks

    @property
    def step(self) -> float:
        return STEPS[self.step_i]

    def move(self, pan_dir: int = 0, tilt_dir: int = 0) -> None:
        """One step in a direction: pan_dir +1 = right, tilt_dir +1 = up."""
        if self.busy:
            return
        self.msg = ""
        self.pt.nudge(dpan=pan_dir * self.step, dtilt=tilt_dir * self.step)

    def bigger_step(self) -> None:
        self.step_i = min(self.step_i + 1, len(STEPS) - 1)

    def smaller_step(self) -> None:
        self.step_i = max(self.step_i - 1, 0)

    def center(self) -> None:
        if not self.busy:
            self.pt.center()

    def release(self) -> None:
        self.pt.release()
        self.msg = "Released servos. The next move re-engages."

    def mark(self) -> None:
        if not self.teach or self.busy:
            return
        if len(self.corners) < 4:
            self.corners.append(self.pt.position)
            self.msg = f"marked corner {len(self.corners)}"
        else:
            self.msg = "Already have 4. Undo one if you want to change a corner."

    def undo(self) -> None:
        if self.teach and self.corners and not self.busy:
            self.corners.pop()
            self.msg = "Removed the last corner"

    def revisit(self) -> None:
        """Drive to each mark in turn so you can check it in the picture."""
        if self.teach and self.corners and not self.busy:
            self.busy = True
            threading.Thread(target=self._revisit_run, daemon=True).start()

    def _revisit_run(self, wait=time.sleep) -> None:
        try:
            for cp, ct in list(self.corners):
                self.pt.move_to(pan=cp, tilt=ct, ease=True)
                wait(1.5)
        finally:
            self.busy = False

    def save(self) -> bool:
        if not self.teach or self.busy:
            return False
        try:
            save_corners(self.corners_file, self.corners)
        except ValueError as e:
            self.msg = f"not saved: {e}"
            return False
        self.saved = True
        return True

    def status_text(self) -> str:
        p, t = self.pt.position
        lines = [f"pan  = {p:6.1f}", f"tilt = {t:6.1f}", f"step = {self.step} deg", ""]
        if self.teach:
            for i, (cp, ct) in enumerate(self.corners, 1):
                lines.append(f"corner {i}: {cp:6.1f} {ct:6.1f}")
            if len(self.corners) < 4:
                lines.append(f"aim at corner {len(self.corners) + 1} of 4, then Mark")
            else:
                lines.append("all 4 marked. Save or Undo the last")
            lines.append("")
        lines.append(self.msg)
        return "\n".join(lines)


def run_gui(cfg: Config, *, teach: bool, crop: float, corners_file: str | Path,
            step: float = 2, pt=None, camera=None, extra=None, on_tick=None,
            title: str | None = None) -> int:
    """Open the window. Pass pt / camera to reuse ones the caller already has open
    (`pantilt live` does, so the classifier keeps reading the same camera).
    extra() returns more text for the readout; on_tick() runs with every frame.
    """
    try:
        import tkinter as tk
    except ImportError:
        print("tkinter is missing.")
        return 1
    from io import BytesIO

    from PIL import Image

    from .cli import _Camera

    try:
        root = tk.Tk()
    except tk.TclError as e:
        print(f"can't open a window ({e}). On the Pi, use its own monitor, or ssh -X on a remote machine.")
        return 1
    root.title(title or ("Teach the board corners" if teach else "Pan-tilt control"))
    own_camera = camera is None
    if own_camera:
        try:
            camera = _Camera(cfg, preview=False, port=None, status=None)
        except Exception as e:
            root.destroy()
            print(f"Couldn't start the camera: {e}")
            return 1

    cw = 960
    ch = round(cw * cfg.camera.preview_height / cfg.camera.preview_width)
    box = crop * ch  # the red box is a square, crop x the frame height

    with (PanTilt(cfg) if pt is None else nullcontext(pt)) as pt:
        pt.center(smooth=False)
        session = AimSession(pt, teach, corners_file, step)

        # The picture, with the green crosshair and the red box the model judges.
        canvas = tk.Canvas(root, width=cw, height=ch, bg="black", highlightthickness=0)
        canvas.grid(row=0, column=0)
        picture = canvas.create_image(0, 0, anchor="nw")
        x, y = cw / 2, ch / 2
        canvas.create_line(x - 15, y, x + 15, y, fill="#00ff78")
        canvas.create_line(x, y - 15, x, y + 15, fill="#00ff78")
        canvas.create_rectangle(x - box / 2, y - box / 2, x + box / 2, y + box / 2,
                                outline="#ff4646", width=2)

        side = tk.Frame(root, padx=10, pady=10)
        side.grid(row=0, column=1, sticky="n")
        readout = tk.Label(side, justify="left", anchor="nw", font=("Menlo", 12), width=34,
                           height=24 if extra else 14)
        readout.pack(anchor="w")

        def save():
            if session.save():
                root.destroy()

        # One table drives both the buttons and the keys. The first key shows on the button.
        actions = [
            ("Center", "c", session.center),
            ("Release servos", "r", session.release),
            ("Smaller step", "-_", session.smaller_step),
            ("Bigger step", "+=", session.bigger_step),
        ]
        if teach:
            actions += [
                ("Mark corner", "m ", session.mark),
                ("Undo last", "u", session.undo),
                ("Go to the marks", "g", session.revisit),
                ("Save and finish", "s", save),
            ]
        actions.append(("Quit", "q", root.destroy))

        keys = {
            "Up": lambda: session.move(tilt_dir=+1), "Down": lambda: session.move(tilt_dir=-1),
            "Right": lambda: session.move(pan_dir=+1), "Left": lambda: session.move(pan_dir=-1),
            "Escape": root.destroy,
        }
        for label, chars, action in actions:
            tk.Button(side, text=f"{label} ({chars[0]})", command=action).pack(fill="x", pady=1)
            keys.update(dict.fromkeys(chars, action))
        tk.Label(side, text="use arrow keys to move", fg="gray").pack(pady=(8, 0))

        def on_key(event):
            action = keys.get(event.keysym) or keys.get(event.char.lower())
            if action:
                action()

        root.bind("<Key>", on_key)
        root.focus_force()

        def tick():
            try:
                frame = camera.grab()[:, :, :3]  # drop the unused 4th channel if present
                buf = BytesIO()
                Image.fromarray(frame).resize((cw, ch)).save(buf, "PPM")
                canvas.photo = tk.PhotoImage(data=buf.getvalue())  # kept so tkinter doesn't drop it
                canvas.itemconfig(picture, image=canvas.photo)
            except Exception as e:
                session.msg = f"camera: {e}"
            text = session.status_text()
            try:
                if on_tick:
                    on_tick()
                if extra:
                    text = f"{extra()}\n\n{text}"
            except Exception as e:  # e.g. the classifier thread died: stop, report it below
                failure.append(e)
                root.destroy()
                return
            readout.config(text=text)
            root.after(40, tick)

        failure: list = []
        tick()
        try:
            root.mainloop()
        finally:
            if own_camera:
                camera.close()
        if failure:
            raise failure[0]
        if teach:
            pt.center()

    if not teach:
        return 0
    if session.saved:
        print(f"saved {len(session.corners)} corners to {session.corners_file}")
        print("next: pantilt board-scan --crop 0.2 --dry-run")
        return 0
    print("quit without saving")
    return 1
