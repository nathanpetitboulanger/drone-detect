"""Lecture des fichiers de labels MATLAB (`groundTruth`, Video Labeler) du jeu Halmstad.

Les fichiers `*_LABELS.mat` contiennent un objet MCOS que `scipy.io.loadmat` ne sait pas
décoder : scipy expose seulement le flux binaire brut `__function_workspace__`. On parcourt
ce flux MAT5 (éléments imbriqués miMATRIX) et on y cherche la table des labels.

Structure observée (vérifiée sur les 285 vidéos visibles) : un tableau dont les éléments
vont par groupes de 1 + C, un groupe par frame vidéo :
  - 1 référence d'objet (classe uint32, dims [6, 1]) ;
  - C cellules de classe double, une par label (ordre alphabétique AIRPLANE, BIRD, DRONE,
    HELICOPTER), de dims [0, 0] (absent) ou [k, 4] (k boîtes `x, y, w, h`, convention
    MATLAB : coin haut-gauche en coordonnées 1-indexées).

Note : `mcos-decoder` (PyPI) suppose une boîte par frame et classe les frames à 2 drones
(cellule [2, 4]) comme vides ; il ajoute aussi une entrée parasite (l'en-tête « DRONE »).
D'où ce décodeur maison.

Usage (test) :
    .venv/bin/python scripts/data/matlab_gt.py data/raw/halmstad/Data/Video_V/V_DRONE_106_LABELS.mat
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass, field

import numpy as np
import scipy.io

LABELS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")
NUMERIC_DTYPES = {1: "<i1", 2: "<u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4",
                  7: "<f4", 9: "<f8", 12: "<i8", 13: "<u8"}
MI_MATRIX = 14
MX_DOUBLE_CLASS, MX_UINT32_CLASS = 6, 13


@dataclass
class Element:
    type_id: int
    payload: bytes = b""
    children: list["Element"] = field(default_factory=list)

    def array(self) -> np.ndarray | None:
        dtype = NUMERIC_DTYPES.get(self.type_id)
        return None if dtype is None else np.frombuffer(self.payload, dtype=dtype)


def parse_stream(buf: bytes, depth: int = 0, max_depth: int = 16) -> list[Element]:
    """Découpe un flux MAT5 en éléments (récursif sur les miMATRIX)."""
    out, off, n = [], 0, len(buf)
    while off + 8 <= n and depth <= max_depth:
        small_type, small_size = struct.unpack_from("<HH", buf, off)
        if small_size != 0 and small_type < 20:  # « small data element » : tag sur 4 octets
            t, size, start = small_type, small_size, off + 4
        else:
            t, size = struct.unpack_from("<II", buf, off)
            start = off + 8
        if t == 0 or t > 20 or start + size > n:
            break
        payload = buf[start:start + size]
        if t == MI_MATRIX:
            out.append(Element(t, children=parse_stream(payload, depth + 1, max_depth)))
        else:
            out.append(Element(t, payload))
        nxt = start + size
        nxt += (8 - nxt % 8) % 8
        if nxt <= off:
            break
        off = nxt
    return out


def matrix_info(e: Element) -> tuple[int, list[int], np.ndarray | None] | None:
    """(classe MATLAB, dims, données) d'un miMATRIX numérique, sinon None."""
    if e.type_id != MI_MATRIX or len(e.children) < 3:
        return None
    flags, dims = e.children[0].array(), e.children[1].array()
    if flags is None or dims is None or len(flags) < 1:
        return None
    data = e.children[3].array() if len(e.children) > 3 else np.empty(0)
    return int(flags[0] & 0xFF), dims.tolist(), data


def is_label_cell(e: Element) -> bool:
    info = matrix_info(e)
    if info is None:
        return False
    cls, dims, _ = info
    return cls == MX_DOUBLE_CLASS and len(dims) == 2 and (dims == [0, 0] or (dims[1] == 4 and dims[0] >= 1))


def is_ref_cell(e: Element) -> bool:
    info = matrix_info(e)
    return info is not None and info[0] == MX_UINT32_CLASS


def find_label_table(elements: list[Element], n_labels: int) -> list[Element] | None:
    """Cherche (en profondeur) le tableau dont les enfants sont [ref, cell × n_labels] × n_frames."""
    stack = list(elements)
    candidates = []
    while stack:
        e = stack.pop()
        if e.type_id != MI_MATRIX:
            continue
        mats = [c for c in e.children if c.type_id == MI_MATRIX]
        g = n_labels + 1
        if len(mats) >= g and len(mats) % g == 0:
            groups = [mats[i:i + g] for i in range(0, len(mats), g)]
            if all(is_ref_cell(gr[0]) and all(is_label_cell(c) for c in gr[1:]) for gr in groups):
                candidates.append(mats)
                continue
        stack.extend(e.children)
    if len(candidates) != 1:
        return None
    return candidates[0]


def boxes_of(cell: Element) -> np.ndarray:
    """Boîtes (k, 4) `x, y, w, h` en pixels, origine 0 (MATLAB est 1-indexé)."""
    info = matrix_info(cell)
    assert info is not None
    _, dims, data = info
    if dims == [0, 0] or data is None or data.size == 0:
        return np.zeros((0, 4))
    k = dims[0]
    boxes = data.astype(np.float64).reshape(4, k).T.copy()  # stockage colonne-major MATLAB
    boxes[:, :2] -= 1.0
    return boxes


def load_halmstad_labels(mat_path: str, labels: tuple[str, ...] = LABELS) -> list[dict[str, np.ndarray]]:
    """Retourne, pour chaque frame, {label: boîtes (k, 4)}. Lève ValueError si format inconnu."""
    raw = scipy.io.loadmat(mat_path)
    fw = raw.get("__function_workspace__")
    if fw is None:
        raise ValueError(f"{mat_path} : pas de __function_workspace__ (pas un objet MCOS)")
    elements = parse_stream(fw.tobytes()[8:])
    table = find_label_table(elements, len(labels))
    if table is None:
        raise ValueError(f"{mat_path} : table de labels introuvable ou ambiguë")
    g = len(labels) + 1
    frames = []
    for i in range(0, len(table), g):
        frames.append({lab: boxes_of(c) for lab, c in zip(labels, table[i + 1:i + g])})
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mat", nargs="+", help="fichiers *_LABELS.mat")
    args = parser.parse_args()
    for path in args.mat:
        frames = load_halmstad_labels(path)
        counts = {lab: sum(len(f[lab]) for f in frames) for lab in LABELS}
        multi = sum(any(len(b) > 1 for b in f.values()) for f in frames)
        print(f"{path}: {len(frames)} frames, boîtes {counts}, frames multi-objets {multi}")


if __name__ == "__main__":
    main()
