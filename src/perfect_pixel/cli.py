import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None


def _parse_grid_size(text):
    if text is None:
        return None
    value = text.lower().replace("x", ",")
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid size must be like 32x32 or 32,32")
    try:
        w = int(parts[0])
        h = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid size must be integers") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("grid size must be positive")
    return (w, h)


def _load_image(path, prefer_cv2=True):
    if prefer_cv2 and cv2 is not None:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if Image is None:
        raise RuntimeError("No image reader available. Install opencv-python or pillow.")
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _save_image(path, rgb):
    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Failed to write image: {path}")
        return
    if Image is None:
        raise RuntimeError("No image writer available. Install opencv-python or pillow.")
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    img.save(path)


def _resolve_backend(name):
    name = name.lower()
    if name == "auto":
        from . import get_perfect_pixel as fn
        return fn, (cv2 is not None)
    if name == "opencv":
        if cv2 is None:
            raise RuntimeError("OpenCV backend selected but opencv-python is not installed.")
        from .perfect_pixel import get_perfect_pixel as fn
        return fn, True
    if name == "numpy":
        from .perfect_pixel_noCV2 import get_perfect_pixel as fn
        return fn, False
    raise RuntimeError(f"Unknown backend: {name}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Perfect Pixel CLI - auto detect grid and refine pixel art."
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path (default: <input>_perfect.png)")
    parser.add_argument("--backend", choices=["auto", "opencv", "numpy"], default="auto")
    parser.add_argument("--sample-method", choices=["center", "majority"], default="center")
    parser.add_argument("--grid-size", type=_parse_grid_size, help="Override grid size, e.g. 32x32")
    parser.add_argument("--min-size", type=float, default=4.0)
    parser.add_argument("--peak-width", type=int, default=6)
    parser.add_argument("--refine-intensity", type=float, default=0.25)
    parser.add_argument("--no-fix-square", action="store_false", dest="fix_square", default=True)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_perfect.png")

    get_perfect_pixel, prefer_cv2 = _resolve_backend(args.backend)
    rgb = _load_image(input_path, prefer_cv2=prefer_cv2)

    w, h, out = get_perfect_pixel(
        rgb,
        sample_method=args.sample_method,
        grid_size=args.grid_size,
        min_size=args.min_size,
        peak_width=args.peak_width,
        refine_intensity=args.refine_intensity,
        fix_square=args.fix_square,
        debug=args.debug,
    )

    if w is None or h is None:
        raise RuntimeError("Failed to refine image.")

    _save_image(output_path, out)
    print(f"Saved: {output_path} ({w}x{h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
