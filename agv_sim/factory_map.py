"""Factory floor map loaded from an image (PNG/JPG).

Dark pixels (grey < obstacle_threshold) are obstacles: walls, racks, machines, pillars.
Anything lighter is free floor. Area outside the image counts as obstacle.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from .config import MapConfig


class FactoryMap:
    def __init__(self, image_path: str = "", meters_per_pixel: float = 0.02,
                 obstacle_threshold: int = 100, blank_size_m=(30.0, 20.0)):
        self.mpp = float(meters_per_pixel)
        self.threshold = int(obstacle_threshold)
        self.image_path = image_path
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Map image not found: {image_path}")
            img = Image.open(image_path)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            self.rgb = np.asarray(img.convert("RGB"))
            self.gray = np.asarray(img.convert("L"))
        else:
            w = int(round(blank_size_m[0] / self.mpp))
            h = int(round(blank_size_m[1] / self.mpp))
            self.rgb = np.full((h, w, 3), 255, np.uint8)
            self.gray = np.full((h, w), 255, np.uint8)
        self.h, self.w = self.gray.shape
        self.obstacle = self.gray < self.threshold

    @classmethod
    def from_config(cls, mc: MapConfig, blocked=None) -> "FactoryMap":
        m = cls(mc.image, mc.meters_per_pixel, mc.obstacle_threshold)
        if blocked:
            m.block(blocked)
        return m

    @classmethod
    def for_route(cls, route) -> "FactoryMap":
        """Floor plan of a route incl. its no-go rectangles."""
        return cls.from_config(route.map, getattr(route, "blocked", None))

    def block(self, rects):
        """Mark rectangles [x0, y0, x1, y1] (m) as obstacles (no-go zones drawn in the app)."""
        self.obstacle = self.obstacle.copy()
        self.blocked = [list(map(float, r)) for r in rects]
        for x0, y0, x1, y1 in self.blocked:
            xa, xb = sorted((x0, x1))
            ya, yb = sorted((y0, y1))
            c0 = max(0, int(xa / self.mpp))
            c1 = min(self.w, int(math.ceil(xb / self.mpp)))
            r0 = max(0, int(self.h - yb / self.mpp))
            r1 = min(self.h, int(math.ceil(self.h - ya / self.mpp)))
            if c1 > c0 and r1 > r0:
                self.obstacle[r0:r1, c0:c1] = True

    # ------------------------------------------------------------------
    @property
    def width_m(self) -> float:
        return self.w * self.mpp

    @property
    def height_m(self) -> float:
        return self.h * self.mpp

    @property
    def extent(self):
        """For matplotlib imshow(origin='upper')."""
        return [0.0, self.width_m, 0.0, self.height_m]

    def set_scale(self, meters_per_pixel: float) -> None:
        self.mpp = float(meters_per_pixel)

    def to_pixel(self, x, y):
        """World metres -> integer (col, row) indices."""
        col = np.floor(np.asarray(x) / self.mpp).astype(int)
        row = (self.h - 1 - np.floor(np.asarray(y) / self.mpp)).astype(int)
        return col, row

    def to_pixel_float(self, x, y):
        return np.asarray(x) / self.mpp, self.h - np.asarray(y) / self.mpp

    def is_obstacle(self, x, y) -> np.ndarray:
        col, row = self.to_pixel(x, y)
        inside = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        out = np.ones(col.shape, dtype=bool)
        out[inside] = self.obstacle[row[inside], col[inside]]
        return out
