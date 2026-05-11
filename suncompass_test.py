import glob
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

from suncompass import SunCompass

EVERY_X = 1
RGB_DIR = "/home/juli/Pictures/test"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")

suncompass = SunCompass()
suncompass.set_eval(dropout=False)

image_paths = []
for pattern in IMAGE_EXTENSIONS:
    image_paths.extend(glob.glob(os.path.join(RGB_DIR, pattern)))
image_paths = sorted(set(image_paths))
image_paths = image_paths[::EVERY_X]

results = []
for path in image_paths:
    image = cv2.imread(path)
    if image is None:
        print(f"Skipping unreadable image: {path}")
        continue
    img_with_suncompass, theta_rad = suncompass.predict_and_draw(image)
    frame_name = os.path.basename(path)
    results.append((frame_name, np.degrees(theta_rad), img_with_suncompass))
    print(f"{frame_name}: {np.degrees(theta_rad):.1f}°")

# Plot all results as a grid
n = len(results)
if n == 0:
    raise ValueError(
        f"No readable images found in '{RGB_DIR}' with extensions: {', '.join(IMAGE_EXTENSIONS)}"
    )

cols = min(5, n)
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
axes = np.atleast_1d(axes).flatten()

for i, (frame_name, theta_deg, img_vis) in enumerate(results):
    axes[i].imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"{frame_name}\n{theta_deg:.1f}°", fontsize=8)
    axes[i].axis("off")

for i in range(n, len(axes)):
    axes[i].axis("off")

plt.suptitle(f"SunCompass predictions (every {EVERY_X} frames)", fontsize=14)
plt.tight_layout()
plt.show()
