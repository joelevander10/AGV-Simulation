"""Generate a demo factory floor PNG (24 x 14 m at 20 mm/pixel) for testing."""
import os
from PIL import Image, ImageDraw

MPP = 0.02
W_M, H_M = 24.0, 14.0
w, h = int(W_M / MPP), int(H_M / MPP)


def box(dr, x0, y0, x1, y1, fill):
    """Rectangle in metres (world frame, origin lower-left)."""
    dr.rectangle([x0 / MPP, h - y1 / MPP, x1 / MPP, h - y0 / MPP], fill=fill)


img = Image.new("RGB", (w, h), (245, 245, 245))
dr = ImageDraw.Draw(img)
# floor markings / zones (light = drivable)
box(dr, 4.5, 2.2, 7.5, 3.8, (200, 230, 200))      # home / loading zone
box(dr, 10.5, 7.0, 13.5, 8.5, (200, 215, 240))    # station A
# walls (black)
box(dr, 0, 0, W_M, 0.2, (0, 0, 0)); box(dr, 0, H_M - 0.2, W_M, H_M, (0, 0, 0))
box(dr, 0, 0, 0.2, H_M, (0, 0, 0)); box(dr, W_M - 0.2, 0, W_M, H_M, (0, 0, 0))
# rack rows (dark grey = obstacle)
box(dr, 4.0, 5.3, 18.0, 6.4, (60, 60, 60))
box(dr, 4.0, 9.1, 18.0, 10.2, (60, 60, 60))
# machines / pillars
box(dr, 20.0, 11.0, 23.0, 13.0, (40, 40, 90))
for x in (8.0, 12.0, 16.0):
    box(dr, x, 12.2, x + 0.4, 12.6, (30, 30, 30))
box(dr, 22.6, 5.6, 23.8, 6.6, (30, 30, 30))        # column near the right turn
out = os.path.join(os.path.dirname(__file__), "..", "maps", "demo_factory.png")
img.save(out)
print("saved", os.path.abspath(out), f"({w}x{h} px, {MPP} m/px)")
