"""Contrôle visuel des labels : dessine les boîtes YOLO sur des images et assemble des planches.

Deux modes :
  - échantillon aléatoire d'images (`--n`, graine fixe), avec drones en priorité ;
  - `--per-sequence` : une vignette par séquence (frame médiane), pour choisir les séquences démo.

Sortie : outputs/data_viz/<nom>.jpg (planches 4 colonnes, vignettes 480 px de large).

Usage :
    .venv/bin/python scripts/data/viz.py --source halmstad --n 24
    .venv/bin/python scripts/data/viz.py --source dut_track --per-sequence
    .venv/bin/python scripts/data/viz.py --images data/yolo/all/images/halmstad/halmstad__V_DRONE_106__000000.jpg
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, add_yolo_dir_arg, layout_from, read_manifests, read_yolo_label

OUT = ROOT / "outputs" / "data_viz"
THUMB_W, COLS = 480, 4


def draw(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    for _, cx, cy, bw, bh in read_yolo_label(lbl):
        x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), max(1, w // 640))
        # repère agrandi pour les petites cibles
        cv2.circle(img, ((x0 + x1) // 2, (y0 + y1) // 2), max(12, (x1 - x0)), (0, 255, 255), max(1, w // 640))
    scale = THUMB_W / w
    thumb = cv2.resize(img, (THUMB_W, int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.putText(thumb, img_path.stem[-40:], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


def sheet(paths: list[Path], out: Path) -> None:
    thumbs = [draw(p) for p in paths]
    th = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in thumbs]
    while len(thumbs) % COLS:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + COLS]) for i in range(0, len(thumbs), COLS)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(paths)} images → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", help="source à visualiser (manifeste)")
    parser.add_argument("--n", type=int, default=24, help="nombre d'images (mode échantillon)")
    parser.add_argument("--per-sequence", action="store_true", help="une vignette par séquence")
    parser.add_argument("--category", help="filtre sur la catégorie de séquence (drone, bird...)")
    parser.add_argument("--images", nargs="+", type=Path, help="images explicites")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", help="nom de la planche (défaut : déduit)")
    add_yolo_dir_arg(parser)
    args = parser.parse_args()
    layout = layout_from(args.yolo_dir)

    if args.images:
        sheet(args.images, OUT / f"{args.name or 'images'}.jpg")
        return
    rows = read_manifests(layout, [args.source] if args.source else None)
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]
    if args.per_sequence:
        by_seq = defaultdict(list)
        for r in rows:
            by_seq[(r["source"], r["sequence"])].append(r)
        picks = [sorted(v, key=lambda r: int(r["frame"]))[len(v) // 2] for _, v in sorted(by_seq.items())]
        name = args.name or f"{args.source or 'all'}_{args.category or 'all'}_per_sequence"
    else:
        rng = random.Random(args.seed)
        pos = [r for r in rows if int(r["n_boxes"]) > 0]
        neg = [r for r in rows if int(r["n_boxes"]) == 0]
        k_neg = min(len(neg), args.n // 6)
        picks = rng.sample(pos, min(len(pos), args.n - k_neg)) + rng.sample(neg, k_neg)
        name = args.name or f"{args.source or 'all'}_sample"
    for i in range(0, len(picks), 48):  # planches de 48 vignettes max
        suffix = f"_{i // 48}" if len(picks) > 48 else ""
        sheet([layout.all_images / r["image"] for r in picks[i:i + 48]], OUT / f"{name}{suffix}.jpg")


if __name__ == "__main__":
    main()
