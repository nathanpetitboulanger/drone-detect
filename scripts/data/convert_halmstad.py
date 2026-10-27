"""Conversion du jeu Halmstad (vidéos visibles, labels MATLAB) au format YOLO, classe unique `drone`.

Entrée  : data/raw/halmstad/Data/Video_V/V_<CLASSE>_<NNN>.mp4 + _LABELS.mat (download.py)
Sortie  : <yolo-dir>/all/{images,labels}/halmstad/ + <yolo-dir>/manifests/halmstad.csv

Choix :
  - vidéos DRONE (30 fps, ~300 frames) : 1 frame sur `--step-drone` (5 par défaut → 6 img/s),
    les frames sans drone de ces vidéos sont gardées comme négatifs (label vide) ;
  - vidéos AIRPLANE / BIRD / HELICOPTER : distracteurs, 1 frame sur `--step-other` (30 → 1 img/s),
    label vide (sauf si un drone y est annoté, ce qui n'arrive pas dans la v1.0.0) ;
  - les vidéos sont regroupées par blocs de `--group-size` numéros consécutifs d'une même classe
    (colonne `group`) : des clips voisins viennent souvent du même vol, ils iront dans le même split.

Usage :
    .venv/bin/python scripts/data/convert_halmstad.py [--step-drone 5] [--step-other 30]
"""

from __future__ import annotations

import argparse
import re
import shutil
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from common import RAW, Layout, add_yolo_dir_arg, layout_from, stem, write_manifest, write_yolo_label
from matlab_gt import load_halmstad_labels

SOURCE = "halmstad"
VIDEO_DIR = RAW / SOURCE / "Data" / "Video_V"
DESC_XLSX = RAW / SOURCE / "Data" / "Video_dataset_description.xlsx"
NAME_RE = re.compile(r"V_([A-Z]+)_(\d+)$")


def load_descriptions() -> dict[str, str]:
    """Tags par vidéo (distance + type de drone / source internet) depuis le fichier Excel."""
    if not DESC_XLSX.exists():
        return {}
    df = pd.read_excel(DESC_XLSX)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SENSOR"].str.strip() == "V"]
    tags = {}
    for _, r in df.iterrows():
        name = f"V_{r['CLASS'].strip()}_{int(r['NUMBER']):03d}"
        parts = [str(r["DISTANCE  BIN"] if "DISTANCE  BIN" in r else r.get("DISTANCE BIN", "")).strip().lower()]
        src = r.get("DRONE TYPE / INTERNET SOURCE")
        if isinstance(src, str) and src.strip():
            parts.append("internet" if src.startswith("http") else src.strip().replace(",", " +"))
        tags[name] = ";".join(p for p in parts if p)
    return tags


def convert_video(task: tuple) -> tuple[list[dict], str | None]:
    video, step_drone, step_other, group_size, quality, tags, layout = task
    name = video.stem
    m = NAME_RE.match(name)
    assert m, name
    category, number = m.group(1).lower(), int(m.group(2))
    try:
        frames_gt = load_halmstad_labels(str(video.with_name(f"{name}_LABELS.mat")))
    except ValueError as e:
        return [], f"{name} ignorée : {e}"
    step = step_drone if category == "drone" else step_other
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, idx = [], -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        if idx >= len(frames_gt):
            break
        ok, img = cap.retrieve()
        if not ok:
            break
        h, w = img.shape[:2]
        fname = stem(SOURCE, name, idx)
        ipath = layout.image(SOURCE, fname)
        if not ipath.exists():
            ipath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ipath), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        boxes = frames_gt[idx].get("DRONE", np.zeros((0, 4)))
        n = write_yolo_label(layout.label(SOURCE, fname), boxes, w, h)
        rows.append({
            "image": f"{SOURCE}/{fname}.jpg", "source": SOURCE, "sequence": name,
            "group": f"{category}_{(number - 1) // group_size:02d}", "frame": idx, "n_boxes": n,
            "width": w, "height": h, "category": category, "tags": tags.get(name, ""), "fps": f"{fps:g}",
        })
    cap.release()
    warn = None
    if idx + 1 != len(frames_gt):
        warn = f"{name} : {idx + 1} frames vidéo vs {len(frames_gt)} frames labellisées"
    return rows, warn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-drone", type=int, default=5, help="1 frame sur N pour les vidéos DRONE")
    parser.add_argument("--step-other", type=int, default=30, help="1 frame sur N pour les vidéos distractrices")
    parser.add_argument("--group-size", type=int, default=5, help="clips consécutifs regroupés (anti-fuite)")
    parser.add_argument("--quality", type=int, default=95, help="qualité JPEG")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clean", action="store_true", help="supprime d'abord les sorties halmstad existantes")
    add_yolo_dir_arg(parser)
    args = parser.parse_args()
    layout: Layout = layout_from(args.yolo_dir)

    if not VIDEO_DIR.exists():
        raise SystemExit(f"{VIDEO_DIR} absent : lancer d'abord scripts/data/download.py --sources halmstad")
    if args.clean:
        for d in (layout.all_images, layout.all_labels):
            shutil.rmtree(d / SOURCE, ignore_errors=True)

    tags = load_descriptions()
    videos = sorted(VIDEO_DIR.glob("V_*.mp4"))
    tasks = [(v, args.step_drone, args.step_other, args.group_size, args.quality, tags, layout) for v in videos]
    rows, warnings = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        for r, warn in ex.map(convert_video, tasks):
            rows.extend(r)
            if warn:
                warnings.append(warn)
    out = write_manifest(layout, SOURCE, rows)

    n_pos = sum(int(r["n_boxes"]) > 0 for r in rows)
    print(f"{len(videos)} vidéos, {len(rows)} images ({n_pos} avec drone, {len(rows) - n_pos} négatives), "
          f"{sum(int(r['n_boxes']) for r in rows)} boîtes → {out}")
    for w in warnings:
        print("  ATTENTION", w)


if __name__ == "__main__":
    main()
