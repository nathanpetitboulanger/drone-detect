"""Constantes et fonctions partagées par les scripts de données (chemins, labels YOLO, manifestes).

Organisation produite sous un dossier racine `--yolo-dir` (défaut data/yolo) :
    <yolo-dir>/all/images/<source>/<source>__<sequence>__<frame>.jpg   (toutes les images converties)
    <yolo-dir>/all/labels/<source>/<source>__<sequence>__<frame>.txt   (labels YOLO, vide = négatif)
    <yolo-dir>/manifests/<source>.csv                                  (1 ligne par image)
    <yolo-dir>/{train,val,test}/{images,labels}/                       (liens durs, créés par split.py)
    <yolo-dir>/splits.csv                                              (séquence → split)

Ce module n'est pas un script : il est importé par convert_*.py, split.py, stats.py, make_video.py, viz.py.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
DEMO = DATA / "demo"
DEFAULT_YOLO_DIR = DATA / "yolo"
SPLITS = ("train", "val", "test")

CLASS_ID = 0  # classe unique « drone »
MIN_BOX_PX = 2.0  # boîtes plus petites (après découpe au bord de l'image) ignorées

MANIFEST_FIELDS = [
    "image",     # chemin relatif à <yolo-dir>/all/images
    "source",    # halmstad, dut_track, dut_det
    "sequence",  # identifiant de séquence (vidéo) dans la source
    "group",     # groupe de séquences à garder dans le même split (anti-fuite)
    "frame",     # index de frame dans la séquence (0-indexé)
    "n_boxes",   # nombre de drones annotés
    "width",
    "height",
    "category",  # drone / bird / airplane / helicopter (catégorie de la séquence)
    "tags",      # attributs libres (distance, origine...)
    "fps",       # FPS d'origine de la séquence (vide si inconnu)
]


@dataclass(frozen=True)
class Layout:
    """Arborescence d'un jeu YOLO construit par ces scripts."""

    root: Path

    @property
    def all_images(self) -> Path:
        return self.root / "all" / "images"

    @property
    def all_labels(self) -> Path:
        return self.root / "all" / "labels"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    def image(self, source: str, name: str) -> Path:
        return self.all_images / source / f"{name}.jpg"

    def label(self, source: str, name: str) -> Path:
        return self.all_labels / source / f"{name}.txt"

    def label_of(self, image_rel: str) -> Path:
        """Label correspondant à une image du manifeste (`<source>/<nom>.jpg`)."""
        return self.all_labels / Path(image_rel).with_suffix(".txt")

    def split_dir(self, split: str) -> Path:
        return self.root / split


def add_yolo_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yolo-dir", type=Path, default=DEFAULT_YOLO_DIR,
                        help="dossier racine du jeu YOLO (défaut : data/yolo)")


def layout_from(path: Path) -> Layout:
    path = Path(path)
    return Layout(path if path.is_absolute() else (ROOT / path))


def stem(source: str, sequence: str, frame: int) -> str:
    """Nom de fichier traçable `<source>__<sequence>__<frame>` (sans extension)."""
    return f"{source}__{sequence}__{frame:06d}"


def clip_boxes(boxes_xywh: np.ndarray, width: int, height: int) -> np.ndarray:
    """Découpe des boîtes `x, y, w, h` (pixels, origine 0) aux bords de l'image ; retire les minuscules."""
    if len(boxes_xywh) == 0:
        return np.zeros((0, 4))
    b = np.asarray(boxes_xywh, dtype=np.float64).reshape(-1, 4)
    x0 = np.clip(b[:, 0], 0, width)
    y0 = np.clip(b[:, 1], 0, height)
    x1 = np.clip(b[:, 0] + b[:, 2], 0, width)
    y1 = np.clip(b[:, 1] + b[:, 3], 0, height)
    out = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1)
    return out[(out[:, 2] >= MIN_BOX_PX) & (out[:, 3] >= MIN_BOX_PX)]


def write_yolo_label(path: Path, boxes_xywh: np.ndarray, width: int, height: int) -> int:
    """Écrit un fichier label YOLO (`0 cx cy w h` normalisés). Retourne le nombre de boîtes."""
    boxes = clip_boxes(boxes_xywh, width, height)
    lines = []
    for x, y, w, h in boxes:
        cx, cy = (x + w / 2) / width, (y + h / 2) / height
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def write_manifest(layout: Layout, source: str, rows: list[dict]) -> Path:
    layout.manifests.mkdir(parents=True, exist_ok=True)
    out = layout.manifests / f"{source}.csv"
    rows = sorted(rows, key=lambda r: (r["sequence"], int(r["frame"])))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out


def read_manifests(layout: Layout, sources: list[str] | None = None) -> list[dict]:
    rows = []
    for p in sorted(layout.manifests.glob("*.csv")):
        if sources and p.stem not in sources:
            continue
        with p.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def read_yolo_label(path: Path) -> np.ndarray:
    """Lit un label YOLO → tableau (k, 5) `cls cx cy w h` (normalisés)."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5))
    return np.loadtxt(path, ndmin=2)
