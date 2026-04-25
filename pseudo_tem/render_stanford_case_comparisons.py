from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-cases-csv", type=Path, required=True)
    parser.add_argument("--compare-slice-csv", type=Path, required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def slugify_case(volume: str, slice_index: int) -> str:
    return f"{Path(volume).stem}__slice{slice_index:03d}"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def normalize_to_uint8(image: np.ndarray, vmax: float) -> Image.Image:
    scaled = np.clip(image / max(vmax, 1e-12), 0.0, 1.0)
    arr = (scaled * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def error_to_rgb(error: np.ndarray, emax: float) -> Image.Image:
    scaled = np.clip(error / max(emax, 1e-12), 0.0, 1.0)
    r = (255.0 * scaled).round().astype(np.uint8)
    g = (255.0 * np.sqrt(scaled)).round().astype(np.uint8)
    b = np.zeros_like(r, dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def fit_height(image: Image.Image, height: int) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def format_metrics(metrics: dict[str, str], prefix: str) -> str:
    return (
        f"{prefix} "
        f"SSIM {float(metrics[f'{prefix.lower()}_ssim']):.4f}  "
        f"PSNR {float(metrics[f'{prefix.lower()}_psnr']):.2f}  "
        f"NMSE {float(metrics[f'{prefix.lower()}_nmse']):.4f}"
    )


def make_case_image(
    category: str,
    volume: str,
    slice_index: int,
    metrics: dict[str, str],
    arrays_root: Path,
) -> Image.Image:
    slug = slugify_case(volume, slice_index)
    new_data = load_npz(arrays_root / "current_model" / category / f"{slug}.npz")
    old_data = load_npz(arrays_root / "legacy_model" / category / f"{slug}.npz")
    zf_data = load_npz(arrays_root / "zero_filled" / category / f"{slug}.npz")

    target = new_data["target"]
    new_output = new_data["output"]
    old_output = old_data["output"]
    zf_output = zf_data["output"]

    vmax = float(
        max(
            target.max(),
            new_output.max(),
            old_output.max(),
            zf_output.max(),
        )
    )
    old_error = np.abs(old_output - target)
    new_error = np.abs(new_output - target)
    emax = float(np.percentile(np.concatenate([old_error.ravel(), new_error.ravel()]), 99.5))

    panels = [
        ("Target", normalize_to_uint8(target, vmax)),
        ("Zero-filled", normalize_to_uint8(zf_output, vmax)),
        ("Old", normalize_to_uint8(old_output, vmax)),
        ("New", normalize_to_uint8(new_output, vmax)),
        ("|Old-Tgt|", error_to_rgb(old_error, emax)),
        ("|New-Tgt|", error_to_rgb(new_error, emax)),
    ]

    font = ImageFont.load_default()
    title_h = 42
    subtitle_h = 36
    panel_label_h = 18
    img_h = 180
    pad = 12

    fitted = [(label, fit_height(image, img_h)) for label, image in panels]
    widths = [image.width for _, image in fitted]
    total_w = sum(widths) + pad * (len(fitted) + 1)
    total_h = title_h + subtitle_h * 3 + panel_label_h + img_h + pad * 3

    canvas = Image.new("RGB", (total_w, total_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    title = f"{category.upper()}  {Path(volume).stem}  slice {slice_index}"
    delta = (
        f"delta SSIM {float(metrics['delta_ssim_new_minus_old']):+.4f}   "
        f"delta PSNR {float(metrics['delta_psnr_new_minus_old']):+.2f} dB   "
        f"delta NMSE {float(metrics['delta_nmse_new_minus_old']):+.4f}"
    )
    new_line = format_metrics(metrics, "new")
    old_line = format_metrics(metrics, "old")
    zf_line = (
        f"ZF  SSIM {float(metrics['zf_ssim']):.4f}  "
        f"PSNR {float(metrics['zf_psnr']):.2f}  "
        f"NMSE {float(metrics['zf_nmse']):.4f}"
    )
    draw.text((pad, pad), title, fill=(240, 240, 240), font=font)
    draw.text((pad, pad + 16), delta, fill=(210, 210, 210), font=font)
    draw.text((pad, pad + title_h), new_line, fill=(160, 220, 160), font=font)
    draw.text((pad, pad + title_h + 12), old_line, fill=(220, 180, 160), font=font)
    draw.text((pad, pad + title_h + 24), zf_line, fill=(180, 180, 220), font=font)

    x = pad
    y = pad + title_h + subtitle_h * 2 + panel_label_h
    for label, image in fitted:
        draw.text((x, y - panel_label_h), label, fill=(235, 235, 235), font=font)
        canvas.paste(image, (x, y))
        x += image.width + pad

    return canvas


def make_overview(case_images: list[Image.Image], title: str) -> Image.Image:
    font = ImageFont.load_default()
    pad = 16
    header_h = 28
    width = max(image.width for image in case_images) + pad * 2
    height = header_h + pad * (len(case_images) + 2) + sum(image.height for image in case_images)
    canvas = Image.new("RGB", (width, height), color=(10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, fill=(245, 245, 245), font=font)
    y = pad + header_h
    for image in case_images:
        canvas.paste(image, (pad, y))
        y += image.height + pad
    return canvas


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = list(csv.DictReader(args.selected_cases_csv.open()))
    compare_rows = {
        (row["volume"], row["slice"]): row
        for row in csv.DictReader(args.compare_slice_csv.open())
    }

    case_images: dict[str, list[Image.Image]] = {"better": [], "worse": []}

    for row in selected_rows:
        volume = row["volume"]
        slice_index = int(row["slice"])
        metrics = compare_rows[(volume, str(slice_index))]
        image = make_case_image(
            category=row["category"],
            volume=volume,
            slice_index=slice_index,
            metrics=metrics,
            arrays_root=args.array_root,
        )
        category_dir = args.output_dir / row["category"]
        category_dir.mkdir(parents=True, exist_ok=True)
        image.save(category_dir / f"{slugify_case(volume, slice_index)}.png")
        case_images[row["category"]].append(image)

    better_overview = make_overview(
        case_images["better"],
        "Stanford Selected Better Cases: target / zero-filled / old / new / |old-target| / |new-target|",
    )
    worse_overview = make_overview(
        case_images["worse"],
        "Stanford Selected Worse Cases: target / zero-filled / old / new / |old-target| / |new-target|",
    )
    better_overview.save(args.output_dir / "better_cases_overview.png")
    worse_overview.save(args.output_dir / "worse_cases_overview.png")
    print(f"wrote images to {args.output_dir}")


if __name__ == "__main__":
    main()
