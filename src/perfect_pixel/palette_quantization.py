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


def load_gpl_palette(path):
    colors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith("gimp palette") or s.lower().startswith("name:") or s.lower().startswith("columns:"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                r = int(parts[0])
                g = int(parts[1])
                b = int(parts[2])
            except ValueError:
                continue
            colors.append([r, g, b])
    if not colors:
        raise RuntimeError(f"No colors found in palette: {path}")
    return np.array(colors, dtype=np.uint8)


def to_uint8(image):
    if image.dtype == np.uint8:
        return image
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def apply_palette(image, palette, chunk_size=65536, color_space="rgb", mask=None):
    img_u8 = to_uint8(image)
    pal_u8 = palette.astype(np.uint8, copy=False)
    if mask is not None:
        mask = mask.reshape(-1)

    if color_space == "lab":
        if cv2 is None:
            raise RuntimeError("LAB palette mapping requires opencv-python.")
        pal_lab = cv2.cvtColor(pal_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.int32)
        flat = img_u8.reshape(-1, 3)
        out = np.empty_like(flat, dtype=np.uint8)
        for start in range(0, flat.shape[0], chunk_size):
            end = min(start + chunk_size, flat.shape[0])
            if mask is not None and not mask[start:end].any():
                out[start:end] = flat[start:end]
                continue
            block = flat[start:end]
            block_lab = cv2.cvtColor(block.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.int32)
            diff = block_lab[:, None, :] - pal_lab[None, :, :]
            dist = (diff * diff).sum(axis=2, dtype=np.int64)
            idx = dist.argmin(axis=1)
            out[start:end] = pal_u8[idx]
        return out.reshape(image.shape)

    img = img_u8.astype(np.int32, copy=False)
    pal = pal_u8.astype(np.int32, copy=False)
    flat = img.reshape(-1, 3)
    out = np.empty_like(flat, dtype=np.uint8)
    for start in range(0, flat.shape[0], chunk_size):
        end = min(start + chunk_size, flat.shape[0])
        if mask is not None and not mask[start:end].any():
            out[start:end] = flat[start:end]
            continue
        block = flat[start:end][:, None, :]
        diff = block - pal[None, :, :]
        dist = (diff * diff).sum(axis=2, dtype=np.int64)
        idx = dist.argmin(axis=1)
        out[start:end] = pal_u8[idx]
    return out.reshape(image.shape)


def load_image(path, prefer_cv2=True):
    if prefer_cv2 and cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), None
        if img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            alpha = img[:, :, 3]
            return rgb, alpha
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None
    if Image is None:
        raise RuntimeError("No image reader available. Install opencv-python or pillow.")
    with Image.open(path) as img:
        if img.mode in ("RGBA", "LA") or ("A" in img.getbands()):
            rgba = img.convert("RGBA")
            arr = np.array(rgba)
            return arr[:, :, :3], arr[:, :, 3]
        return np.array(img.convert("RGB")), None


def save_image(path, rgb, alpha=None):
    if cv2 is not None:
        if alpha is not None:
            bgra = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGRA)
            bgra[:, :, 3] = alpha
            ok = cv2.imwrite(str(path), bgra)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(str(path), bgr)
        if not ok:
            raise RuntimeError(f"Failed to write image: {path}")
        return
    if Image is None:
        raise RuntimeError("No image writer available. Install opencv-python or pillow.")
    if alpha is not None:
        rgba = np.dstack([rgb.astype(np.uint8), alpha.astype(np.uint8)])
        img = Image.fromarray(rgba, mode="RGBA")
    else:
        img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    img.save(path)


def alpha_from_source(alpha, size, coverage_threshold=0.5):
    if alpha is None:
        return None
    h, w = size
    if alpha.shape[0] == h and alpha.shape[1] == w:
        return np.where(alpha > 0, 255, 0).astype(np.uint8)
    binary = np.where(alpha > 0, 255, 0).astype(np.uint8)
    if cv2 is not None:
        resized = cv2.resize(binary, (w, h), interpolation=cv2.INTER_AREA)
    elif Image is not None:
        resized = np.array(Image.fromarray(binary).resize((w, h), resample=Image.BOX))
    else:
        raise RuntimeError("Alpha resize requires opencv-python or pillow.")
    thr = int(round(255 * float(coverage_threshold)))
    return np.where(resized >= thr, 255, 0).astype(np.uint8)


def resolve_backend(name):
    name = name.lower()
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if name == "auto":
        from perfect_pixel import get_perfect_pixel as fn
        return fn, (cv2 is not None)
    if name == "opencv":
        if cv2 is None:
            raise RuntimeError("OpenCV backend selected but opencv-python is not installed.")
        from perfect_pixel.perfect_pixel import get_perfect_pixel as fn
        return fn, True
    if name == "numpy":
        from perfect_pixel.perfect_pixel_noCV2 import get_perfect_pixel as fn
        return fn, False
    raise RuntimeError(f"Unknown backend: {name}")


def run_cli(args):
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_perfect.png")

    get_perfect_pixel, prefer_cv2 = resolve_backend(args.backend)
    rgb, alpha = load_image(input_path, prefer_cv2=prefer_cv2)

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

    alpha_out = None
    if alpha is not None:
        alpha_out = alpha_from_source(alpha, out.shape[:2], coverage_threshold=args.alpha_coverage)

    if args.palette:
        palette = load_gpl_palette(args.palette)
        palette_space = args.palette_space or ("lab" if cv2 is not None else "rgb")
        mask = None
        if alpha_out is not None:
            mask = (alpha_out > 0).reshape(-1)
        out = apply_palette(out, palette, color_space=palette_space, mask=mask)

    save_image(output_path, out, alpha=alpha_out)
    return w, h, output_path
