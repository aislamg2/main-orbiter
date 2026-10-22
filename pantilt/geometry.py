"""Board geometry: turn positions on the flat map into pan/tilt angles.

`scan.raster` spaces poses evenly in *angle*. On a flat board that is not the
same as spacing them evenly in *millimetres* - tan() is not linear - so the
outer tiles cover more board than the middle ones and a reconstructed mosaic
comes out stretched at the edges. This module lays the grid out in board
millimetres and converts each position to angles on the way out.

The tile footprint is passed in rather than configured, because what the model
actually sees is the centre square of the frame at `[model] crop`. Deriving it
at run time keeps the geometry and the classifier looking at the same patch.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from .config import BoardConfig


@dataclass(frozen=True)
class Tile:
    index: int
    row: int
    col: int
    x_mm: float        # tile centre in board coordinates, origin at board centre
    y_mm: float        # +y is up
    pan_deg: float
    tilt_deg: float


def tile_mm(board: BoardConfig, frame_h: int, frame_w: int, crop: float) -> float:
    """Side length on the board, in mm, of the square the model classifies."""
    side_px = max(1, int(min(frame_h, frame_w) * crop))
    return side_px * board.mm_per_px


def _axis_positions(span_mm: float, tile: float, overlap: float) -> list[float]:
    """Tile centres along one axis, outermost tiles flush with the board edges."""
    if tile >= span_mm:
        return [0.0]
    step = tile * (1 - overlap)
    n = math.ceil((span_mm - tile) / step) + 1
    usable = span_mm - tile
    first = -span_mm / 2 + tile / 2
    return [first + usable * i / (n - 1) for i in range(n)]


def tile_positions(board: BoardConfig, tile: float) -> tuple[list[float], list[float]]:
    return (
        _axis_positions(board.width_mm, tile, board.overlap),
        _axis_positions(board.height_mm, tile, board.overlap),
    )


def xy_to_angles(board: BoardConfig, x_mm: float, y_mm: float) -> tuple[float, float]:
    """Angles that put board point (x_mm, y_mm) in the middle of the frame.

    The hypot() in the tilt term is not cosmetic: on a pan-then-tilt bracket the
    tilt axis has already been swung by the pan, so tilt is measured against the
    slant distance to the target rather than the perpendicular distance to the
    board. Using atan2(y, distance) leaves a few mm of error at the corners,
    which shows up as a sheared mosaic.
    """
    pan_off = math.degrees(math.atan2(x_mm, board.distance_mm))
    tilt_off = math.degrees(math.atan2(y_mm, math.hypot(x_mm, board.distance_mm)))
    return (
        board.pan_center_deg + board.pan_sign * pan_off,
        board.tilt_center_deg + board.tilt_sign * tilt_off,
    )


def angles_to_xy(board: BoardConfig, pan_deg: float, tilt_deg: float) -> tuple[float, float]:
    """Inverse of xy_to_angles - where on the board a pose is looking."""
    pan_off = math.radians((pan_deg - board.pan_center_deg) / board.pan_sign)
    tilt_off = math.radians((tilt_deg - board.tilt_center_deg) / board.tilt_sign)
    x = board.distance_mm * math.tan(pan_off)
    y = math.hypot(x, board.distance_mm) * math.tan(tilt_off)
    return x, y


def scan_order(board: BoardConfig, tile: float, serpentine: bool = False) -> Iterator[Tile]:
    """Tiles in capture order.

    Unidirectional by default: every pose is approached from the same side, so
    gear backlash is a constant offset instead of flipping sign on alternate
    rows. Serpentine halves the travel, which only matters if movement time
    costs you something.
    """
    xs, ys = tile_positions(board, tile)
    i = 0
    for r, y in enumerate(ys):
        row = xs if (not serpentine or r % 2 == 0) else list(reversed(xs))
        for x in row:
            pan, tilt = xy_to_angles(board, x, y)
            yield Tile(i, r, xs.index(x), x, y, pan, tilt)
            i += 1


def describe(board: BoardConfig, tile: float) -> str:
    """Plan summary - worth reading before committing to a scan."""
    xs, ys = tile_positions(board, tile)
    corners = [xy_to_angles(board, x, y) for x in (min(xs), max(xs)) for y in (min(ys), max(ys))]
    pans = [p for p, _ in corners]
    tilts = [t for _, t in corners]
    return "\n".join([
        f"distance     {board.distance_mm:.0f} mm",
        f"scale        {board.mm_per_px:.5f} mm per pixel",
        f"board        {board.width_mm:.0f} x {board.height_mm:.0f} mm",
        f"tile         {tile:.1f} mm square  (overlap {board.overlap:.0%})",
        f"grid         {len(xs)} cols x {len(ys)} rows = {len(xs) * len(ys)} tiles",
        f"pan range    {min(pans):.1f} .. {max(pans):.1f} deg",
        f"tilt range   {min(tilts):.1f} .. {max(tilts):.1f} deg",
    ])
