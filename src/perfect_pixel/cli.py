import argparse
import sys
from pathlib import Path


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


def _ensure_package_import():
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None):
    _ensure_package_import()
    from perfect_pixel import palette_quantization

    parser = argparse.ArgumentParser(
        description="Perfect Pixel CLI - auto detect grid and refine pixel art."
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path (default: <input>_perfect.png)")
    parser.add_argument("--backend", choices=["auto", "opencv", "numpy"], default="auto")
    parser.add_argument("--sample-method", choices=["center", "majority"], default="center")
    parser.add_argument("--grid-size", type=_parse_grid_size, help="Override grid size, e.g. 32x32")
    parser.add_argument("--palette", help="GPL palette path to constrain output colors")
    parser.add_argument("--palette-space", choices=["rgb", "lab"], help="Color space for palette matching")
    parser.add_argument("--alpha-coverage", type=float, default=0.5, help="Opaque coverage needed per output pixel (0-1)")
    parser.add_argument("--min-size", type=float, default=4.0)
    parser.add_argument("--peak-width", type=int, default=6)
    parser.add_argument("--refine-intensity", type=float, default=0.25)
    parser.add_argument("--no-fix-square", action="store_false", dest="fix_square", default=True)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    w, h, output_path = palette_quantization.run_cli(args)
    print(f"Saved: {output_path} ({w}x{h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
