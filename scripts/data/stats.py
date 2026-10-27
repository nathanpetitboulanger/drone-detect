"""Statistiques du jeu YOLO découpé : séquences, images, boîtes, négatifs, tailles de boîtes, résolutions.

Par split (et par source dans chaque split) :
  - nb de séquences, d'images, de boîtes, d'images négatives (label vide) ;
  - taille des boîtes en pixels, côté équivalent √(w·h) (mêmes seuils que COCO small/medium) :
    part < 16 px, < 32 px, < 96 px, médiane ; plus la médiane du plus grand côté ;
  - résolutions d'images.
Vérifie aussi l'absence de séquence partagée entre splits (split.check_leakage).

Sortie : tableaux Markdown à l'écran et dans <yolo-dir>/stats.md (repris dans docs/DATA.md).

Usage :
    .venv/bin/python scripts/data/stats.py
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict

import numpy as np

from common import SPLITS, add_yolo_dir_arg, layout_from, read_manifests, read_yolo_label
from split import check_leakage


def summarize(items: list[dict]) -> dict:
    sides = np.array([s for it in items for s in it["sides"]])
    longs = np.array([s for it in items for s in it["longs"]])
    def pct(t: float) -> str:
        return f"{(sides < t).mean():.0%}" if len(sides) else "-"

    return {
        "séquences": len({(it["source"], it["sequence"]) for it in items}),
        "images": len(items),
        "boîtes": len(sides),
        "négatifs": sum(not it["sides"] for it in items),
        "< 16 px": pct(16),
        "< 32 px": pct(32),
        "< 96 px": pct(96),
        "médiane √(w·h)": f"{np.median(sides):.1f}" if len(sides) else "-",
        "médiane grand côté": f"{np.median(longs):.1f}" if len(longs) else "-",
    }


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_yolo_dir_arg(parser)
    args = parser.parse_args()
    layout = layout_from(args.yolo_dir)

    meta = {os.path.basename(r["image"]): r for r in read_manifests(layout)}
    per_split: dict[str, list[dict]] = defaultdict(list)
    resolutions: dict[str, Counter] = defaultdict(Counter)
    for split in SPLITS:
        for name in sorted(os.listdir(layout.split_dir(split) / "images")):
            r = meta[name]
            w, h = int(r["width"]), int(r["height"])
            lbl = read_yolo_label(layout.split_dir(split) / "labels" / name.replace(".jpg", ".txt"))
            bw, bh = lbl[:, 3] * w, lbl[:, 4] * h
            per_split[split].append({"source": r["source"], "sequence": r["sequence"],
                                     "sides": np.sqrt(bw * bh).tolist(), "longs": np.maximum(bw, bh).tolist()})
            resolutions[split][f"{w}×{h}"] += 1

    out = [f"# Statistiques — `{layout.root.relative_to(layout.root.parents[1])}`", ""]
    header: list[str] = []
    rows = []
    for split in (*SPLITS, "total"):
        items = per_split[split] if split != "total" else [it for s in SPLITS for it in per_split[s]]
        s = summarize(items)
        header = ["split", *s.keys()]
        rows.append([f"**{split}**", *s.values()])
        if split == "total":
            continue
        for source in sorted({it["source"] for it in items}):
            s = summarize([it for it in items if it["source"] == source])
            rows.append([f"{split} / {source}", *s.values()])
    out += ["## Par split et par source", "", md_table(header, rows), ""]

    out += ["## Résolutions d'images", ""]
    res_rows = []
    for split in SPLITS:
        top = resolutions[split].most_common(4)
        others = sum(resolutions[split].values()) - sum(c for _, c in top)
        res_rows.append([split, ", ".join(f"{k} : {c}" for k, c in top) + (f", autres : {others}" if others else "")])
    out += [md_table(["split", "résolution : nb d'images"], res_rows), ""]

    text = "\n".join(out)
    print(text)
    (layout.root / "stats.md").write_text(text)
    print(f"→ {layout.root / 'stats.md'}\n\nVérification des fuites :")
    if not check_leakage(layout):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
