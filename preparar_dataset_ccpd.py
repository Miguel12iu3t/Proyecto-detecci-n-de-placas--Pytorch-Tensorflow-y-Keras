"""
Genera un dataset binario (placa / no_placa) a partir de CCPD2019.

Uso recomendado:
  python preparar_dataset_ccpd.py \
    --ccpd-root "C:/Users/gomez/OneDrive/Escritorio/placas/CCPD2019" \
    --out-dir "dataset" \
    --min-per-folder 500 \
    --target-per-folder 1000 \
    --max-per-folder 2000
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def parse_bbox_from_ccpd_name(file_path: Path) -> Optional[Tuple[int, int, int, int]]:
    """
    Extrae bbox desde el nombre CCPD.
    Formato esperado (fragmento): ...-x1&y1_x2&y2-...
    """
    stem = file_path.stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None

    bbox_part = parts[2]
    if "_" not in bbox_part or "&" not in bbox_part:
        return None

    try:
        p1, p2 = bbox_part.split("_")
        x1, y1 = map(int, p1.split("&"))
        x2, y2 = map(int, p2.split("&"))
    except Exception:
        return None

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return x1, y1, x2, y2


def clip_bbox(bbox: Tuple[int, int, int, int], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return x1, y1, x2, y2


def expand_bbox(bbox: Tuple[int, int, int, int], w: int, h: int, frac: float = 0.06) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    px = int(bw * frac)
    py = int(bh * frac)
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    )


def random_negative_patch(
    img_h: int,
    img_w: int,
    plate_bbox: Tuple[int, int, int, int],
    tries: int = 40,
) -> Optional[Tuple[int, int, int, int]]:
    px1, py1, px2, py2 = plate_bbox
    pw, ph = max(20, px2 - px1), max(10, py2 - py1)

    for _ in range(tries):
        scale = random.uniform(0.7, 1.4)
        nw = int(pw * scale)
        nh = int(ph * scale)
        if nw >= img_w or nh >= img_h:
            continue

        x1 = random.randint(0, img_w - nw)
        y1 = random.randint(0, img_h - nh)
        cand = (x1, y1, x1 + nw, y1 + nh)

        if iou_xyxy(cand, plate_bbox) < 0.03:
            return cand

    return None


def collect_images(folder: Path) -> List[Path]:
    images = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            images.append(p)
    return images


def resolve_sample_count(available: int, min_n: int, target_n: int, max_n: int) -> int:
    target = max(min_n, target_n)
    target = min(target, max_n)
    return min(available, target)


def process_subset(
    subset_dir: Path,
    out_placa: Path,
    out_no_placa: Path,
    min_per_folder: int,
    target_per_folder: int,
    max_per_folder: int,
) -> Tuple[int, int, int]:
    images = collect_images(subset_dir)
    total = len(images)
    if total == 0:
        print(f"[AVISO] Sin imágenes en: {subset_dir}")
        return 0, 0, 0

    n = resolve_sample_count(total, min_per_folder, target_per_folder, max_per_folder)
    chosen = random.sample(images, n) if total > n else images

    ok_pos = 0
    ok_neg = 0
    skipped = 0

    for idx, image_path in enumerate(chosen, start=1):
        img = cv2.imread(str(image_path))
        if img is None:
            skipped += 1
            continue

        h, w = img.shape[:2]
        bbox = parse_bbox_from_ccpd_name(image_path)
        if bbox is None:
            skipped += 1
            continue

        bbox = clip_bbox(bbox, w, h)
        if bbox is None:
            skipped += 1
            continue

        plate_bbox = expand_bbox(bbox, w, h)
        x1, y1, x2, y2 = plate_bbox
        plate_crop = img[y1:y2, x1:x2]
        if plate_crop.size == 0:
            skipped += 1
            continue

        base_name = f"{subset_dir.name}_{idx:06d}"
        pos_path = out_placa / f"{base_name}.jpg"
        cv2.imwrite(str(pos_path), plate_crop)
        ok_pos += 1

        neg_bbox = random_negative_patch(h, w, plate_bbox)
        if neg_bbox is not None:
            nx1, ny1, nx2, ny2 = neg_bbox
            neg_crop = img[ny1:ny2, nx1:nx2]
            if neg_crop.size > 0:
                neg_path = out_no_placa / f"{base_name}.jpg"
                cv2.imwrite(str(neg_path), neg_crop)
                ok_neg += 1

    return ok_pos, ok_neg, skipped


def main():
    parser = argparse.ArgumentParser(description="Preparar dataset placa/no_placa desde CCPD2019")
    parser.add_argument("--ccpd-root", type=str, required=True, help="Ruta a la carpeta CCPD2019")
    parser.add_argument("--out-dir", type=str, default="dataset", help="Carpeta de salida")
    parser.add_argument("--min-per-folder", type=int, default=500, help="Mínimo objetivo por subcarpeta")
    parser.add_argument("--target-per-folder", type=int, default=1000, help="Objetivo por subcarpeta")
    parser.add_argument("--max-per-folder", type=int, default=2000, help="Máximo por subcarpeta")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ccpd_root = Path(args.ccpd_root)
    out_dir = Path(args.out_dir)

    if not ccpd_root.exists() or not ccpd_root.is_dir():
        raise FileNotFoundError(f"Carpeta CCPD inválida: {ccpd_root}")

    out_placa = out_dir / "placa"
    out_no_placa = out_dir / "no_placa"
    out_placa.mkdir(parents=True, exist_ok=True)
    out_no_placa.mkdir(parents=True, exist_ok=True)

    subsets = [p for p in ccpd_root.iterdir() if p.is_dir()]
    subsets.sort(key=lambda p: p.name)
    if not subsets:
        raise ValueError(f"No hay subcarpetas dentro de: {ccpd_root}")

    print("=" * 70)
    print("Preparando dataset desde CCPD")
    print(f"Raíz CCPD: {ccpd_root}")
    print(f"Salida:    {out_dir}")
    print(f"Regla por carpeta: min={args.min_per_folder}, target={args.target_per_folder}, max={args.max_per_folder}")
    print("=" * 70)

    total_pos = 0
    total_neg = 0
    total_skip = 0

    for subset in subsets:
        pos, neg, skipped = process_subset(
            subset_dir=subset,
            out_placa=out_placa,
            out_no_placa=out_no_placa,
            min_per_folder=args.min_per_folder,
            target_per_folder=args.target_per_folder,
            max_per_folder=args.max_per_folder,
        )
        total_pos += pos
        total_neg += neg
        total_skip += skipped
        print(f"[{subset.name}] placa={pos}  no_placa={neg}  omitidas={skipped}")

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print(f"placa:    {total_pos}")
    print(f"no_placa: {total_neg}")
    print(f"omitidas: {total_skip}")
    print(f"Dataset generado en: {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
