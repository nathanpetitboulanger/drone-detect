"""Conversion de DUT Anti-UAV (partie détection, 10 000 images, VOC XML) au format YOLO, classe `drone`.

Entrée  : data/raw/dut_det/{train,val,test}/{img,xml}/NNNNN.{jpg,xml}   (download.py --sources dut_det)
          <yolo-dir>/manifests/dut_track.csv                          (convert_dut_track.py, avant)
Sortie  : <yolo-dir>/all/{images,labels}/dut_det/ + <yolo-dir>/manifests/dut_det.csv

Ces images n'ont pas d'identifiant de séquence, et leur découpage officiel train/val/test fuit :
ce sont des frames de vidéos, ~65 % des images de test ont une quasi-copie dans train (mesuré).
Une partie vient aussi des mêmes vidéos / décors que DUT tracking. Donc :
  - les trois splits officiels sont fusionnés et la source est utilisée **en train seulement**
    (split.py, TRAIN_ONLY) ;
  - chaque image est comparée aux frames DUT tracking (descripteur global « structure » :
    vignette 64×36 en niveaux de gris passe-haut, similarité cosinus) ; si la similarité dépasse
    `--link-threshold` (0,5 : contrôle visuel, plateau net entre 0,5 et 0,8), la colonne `group`
    vaut `dut_track:<groupe>` et split.py écarte l'image si ce groupe est en val/test.

Les JPEG d'origine sont liés en dur (pas de ré-encodage). Boîtes VOC (xmin, ymin, xmax, ymax,
1-indexées) converties comme le script VOC d'Ultralytics.

Usage :
    .venv/bin/python scripts/data/convert_dut_det.py [--link-threshold 0.5]
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import RAW, add_yolo_dir_arg, layout_from, stem, write_manifest, write_yolo_label

SOURCE = "dut_det"
DET_ROOT = RAW / SOURCE
TRACK_ROOT = RAW / "dut_track" / "Anti-UAV-Tracking-V0"
OFFICIAL_SPLITS = ("train", "val", "test")


def structure_descriptor(path: str) -> np.ndarray:
    """Vignette 64×36 passe-haut normalisée : robuste aux ciels uniformes, sensible au décor."""
    im = cv2.imread(path, cv2.IMREAD_REDUCED_GRAYSCALE_4).astype(np.float32)
    t = cv2.resize(im, (64, 36), interpolation=cv2.INTER_AREA)
    v = (t - cv2.GaussianBlur(t, (0, 0), 3)).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def parse_voc(xml_path: Path) -> np.ndarray:
    """Boîtes `x, y, w, h` (pixels, origine 0) d'un fichier VOC."""
    boxes = []
    for obj in ET.parse(xml_path).getroot().findall("object"):
        b = obj.find("bndbox")
        x0, y0, x1, y1 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append([x0 - 1, y0 - 1, x1 - x0, y1 - y0])
    return np.array(boxes).reshape(-1, 4)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--link-threshold", type=float, default=0.5,
                        help="similarité au-delà de laquelle une image est rattachée à une séquence DUT tracking")
    parser.add_argument("--track-step", type=int, default=3, help="1 frame DUT tracking sur N pour la comparaison")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clean", action="store_true", help="supprime d'abord les sorties dut_det existantes")
    add_yolo_dir_arg(parser)
    args = parser.parse_args()
    layout = layout_from(args.yolo_dir)

    if not (DET_ROOT / "test" / "xml").exists():
        raise SystemExit(f"{DET_ROOT} incomplet : lancer d'abord scripts/data/download.py --sources dut_det")
    track_manifest = layout.manifests / "dut_track.csv"
    if not track_manifest.exists():
        raise SystemExit(f"{track_manifest} absent : lancer d'abord convert_dut_track.py (même --yolo-dir)")
    with track_manifest.open() as f:
        track_group = {r["sequence"]: r["group"] for r in csv.DictReader(f)}
    if args.clean:
        for d in (layout.all_images, layout.all_labels):
            shutil.rmtree(d / SOURCE, ignore_errors=True)

    images = [(split, p) for split in OFFICIAL_SPLITS for p in sorted((DET_ROOT / split / "img").glob("*.jpg"))]
    track_frames = [p for seq in sorted(track_group) for p in sorted((TRACK_ROOT / seq).glob("*.jpg"))[::args.track_step]]
    print(f"descripteurs : {len(images)} images détection, {len(track_frames)} frames tracking…")
    with Pool(args.workers) as pool:
        det_feat = np.stack(pool.map(structure_descriptor, [str(p) for _, p in images], chunksize=64))
        trk_feat = np.stack(pool.map(structure_descriptor, [str(p) for p in track_frames], chunksize=64))
    trk_seq = np.array([p.parent.name for p in track_frames])
    best_sim = np.empty(len(images))
    best_seq = np.empty(len(images), dtype=object)
    for i in range(0, len(images), 1024):  # par blocs pour limiter la mémoire
        s = det_feat[i:i + 1024] @ trk_feat.T
        best_sim[i:i + 1024] = s.max(1)
        best_seq[i:i + 1024] = trk_seq[s.argmax(1)]

    rows = []
    for (split, img), sim, seq in zip(images, best_sim, best_seq):
        with Image.open(img) as im:
            w, h = im.size  # taille réelle (l'XML peut différer)
        sequence = f"{split}-{img.stem}"
        name = stem(SOURCE, sequence, 0)
        link_or_copy(img, layout.image(SOURCE, name))
        n = write_yolo_label(layout.label(SOURCE, name), parse_voc(img.parent.parent / "xml" / f"{img.stem}.xml"), w, h)
        linked = sim >= args.link_threshold
        rows.append({
            "image": f"{SOURCE}/{name}.jpg", "source": SOURCE, "sequence": sequence,
            "group": f"dut_track:{track_group[seq]}" if linked else sequence, "frame": 0, "n_boxes": n,
            "width": w, "height": h, "category": "drone",
            "tags": f"officiel={split};proche={seq}:{sim:.2f}", "fps": "",
        })
    out = write_manifest(layout, SOURCE, rows)
    n_pos = sum(r["n_boxes"] > 0 for r in rows)
    n_link = sum(r["group"].startswith("dut_track:") for r in rows)
    print(f"{len(rows)} images ({n_pos} avec drone, {len(rows) - n_pos} négatives), "
          f"{n_link} rattachées à une séquence DUT tracking (sim ≥ {args.link_threshold}) → {out}")


if __name__ == "__main__":
    main()
