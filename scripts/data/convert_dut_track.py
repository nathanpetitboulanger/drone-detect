"""Conversion de DUT Anti-UAV (partie tracking, 20 séquences) au format YOLO, classe unique `drone`.

Entrée  : data/raw/dut_track/Anti-UAV-Tracking-V0/videoNN/NNNNN.jpg
          data/raw/dut_track/Anti-UAV-Tracking-V0GT/videoNN_gt.txt  (1 ligne/frame : `x y w h`,
          pixels, `-100 -100 -100 -100` = cible absente)
Sortie  : <yolo-dir>/all/{images,labels}/dut_track/ + <yolo-dir>/manifests/dut_track.csv

Choix : 1 frame sur `--step` (5 par défaut). Les frames « absentes » sont gardées comme négatifs.
Les JPEG d'origine (720p / 1080p) sont liés en dur (pas de ré-encodage). Chaque séquence est
son propre groupe pour le split, sauf celles filmées visiblement au même endroit (SAME_SCENE),
gardées ensemble. FPS d'origine non documenté par les auteurs (laissé vide).

Usage :
    .venv/bin/python scripts/data/convert_dut_track.py [--step 5] [--clean]
"""

from __future__ import annotations

import argparse
import os
import shutil

import cv2
import numpy as np

from common import RAW, add_yolo_dir_arg, layout_from, stem, write_manifest, write_yolo_label

SOURCE = "dut_track"
IMG_ROOT = RAW / SOURCE / "Anti-UAV-Tracking-V0"
GT_ROOT = RAW / SOURCE / "Anti-UAV-Tracking-V0GT"
# Séquences au même décor (contrôle visuel, outputs/data_viz/dut_track_all_per_sequence.jpg).
SAME_SCENE = {"video08": "video07", "video10": "video09"}


def link_or_copy(src, dst) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, default=5, help="1 frame sur N")
    parser.add_argument("--clean", action="store_true", help="supprime d'abord les sorties dut_track existantes")
    add_yolo_dir_arg(parser)
    args = parser.parse_args()
    layout = layout_from(args.yolo_dir)

    if not GT_ROOT.exists():
        raise SystemExit(f"{GT_ROOT} absent : lancer d'abord scripts/data/download.py --sources dut_track")
    if args.clean:
        for d in (layout.all_images, layout.all_labels):
            shutil.rmtree(d / SOURCE, ignore_errors=True)

    rows = []
    for gt_file in sorted(GT_ROOT.glob("video*_gt.txt")):
        seq = gt_file.stem.replace("_gt", "")
        gt = np.loadtxt(gt_file, ndmin=2)
        frames = sorted((IMG_ROOT / seq).glob("*.jpg"))
        if len(frames) != len(gt):
            print(f"  ATTENTION {seq} : {len(frames)} images vs {len(gt)} lignes de GT, on tronque")
        h, w = cv2.imread(str(frames[0])).shape[:2]  # résolution constante dans une séquence
        for idx in range(0, min(len(frames), len(gt)), args.step):
            box = gt[idx]
            boxes = box[None, :] if (box[2] > 0 and box[3] > 0) else np.zeros((0, 4))
            name = stem(SOURCE, seq, idx)
            link_or_copy(frames[idx], layout.image(SOURCE, name))
            n = write_yolo_label(layout.label(SOURCE, name), boxes, w, h)
            rows.append({
                "image": f"{SOURCE}/{name}.jpg", "source": SOURCE, "sequence": seq, "group": SAME_SCENE.get(seq, seq),
                "frame": idx, "n_boxes": n, "width": w, "height": h, "category": "drone",
                "tags": f"{h}p", "fps": "",
            })
    out = write_manifest(layout, SOURCE, rows)
    n_pos = sum(r["n_boxes"] > 0 for r in rows)
    print(f"{len({r['sequence'] for r in rows})} séquences, {len(rows)} images ({n_pos} avec drone, "
          f"{len(rows) - n_pos} négatives) → {out}")


if __name__ == "__main__":
    main()
