import cv2
import matplotlib.pyplot as plt
from perfectPixel import get_perfect_pixel

bgr = cv2.imread("image.png", cv2.IMREAD_COLOR)
if bgr is None:
    raise FileNotFoundError("Cannot read image: image.png")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

w, h, out = get_perfect_pixel(rgb, debug=True)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Input")
plt.imshow(rgb)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Pixel-perfect ({w}×{h})")
plt.imshow(out)
plt.axis("off")

plt.show()