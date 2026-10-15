"""Scan patterns for sweeping the camera across the map."""

from __future__ import annotations

from collections.abc import Iterator


def linspace(a: float, b: float, n: int) -> list[float]:
    if n < 1:
        raise ValueError("need at least 1 step")
    if n == 1:
        return [(a + b) / 2]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def raster(
    pan_range: tuple[float, float],
    tilt_range: tuple[float, float],
    cols: int,
    rows: int,
) -> Iterator[tuple[float, float]]:
    """Serpentine grid: left->right on one row, right->left on the next.

    Serpentine order avoids a long return swing at the end of every row.
    """
    pans = linspace(*pan_range, cols)
    for r, tilt in enumerate(linspace(*tilt_range, rows)):
        for pan in pans if r % 2 == 0 else reversed(pans):
            yield pan, tilt
