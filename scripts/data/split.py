"""Découpage train/val/test **par séquence** (jamais une séquence dans deux splits).

Lit les manifestes <yolo-dir>/manifests/*.csv (produits par convert_*.py) et répartit les
**groupes** de séquences (colonne `group` : une séquence, ou plusieurs clips d'un même vol / décor)
entre les splits, avec une graine fixe, stratifié par (source, catégorie), en visant ~70/15/15 en
nombre d'images. Les séquences de démo (`DEMO_SEQUENCES`) sont forcées en test.

Sources « train seulement » (`TRAIN_ONLY`, ex. dut_det : images sans notion de séquence) : leurs
images vont en train, sauf celles rattachées (colonne `group` = `<source>:<groupe>`, cf.
convert_dut_det.py) à un groupe d'une autre source placé en val/test, qui sont écartées.
Ces sources n'entrent pas dans le tirage : les splits des autres sources n'en dépendent pas.

Sorties (idempotent : les dossiers de split sont reconstruits à chaque exécution) :
    <yolo-dir>/{train,val,test}/{images,labels}/   liens durs vers <yolo-dir>/all/
    <yolo-dir>/splits.csv                          séquence → split, nb images / positives
Puis vérifie qu'aucune séquence n'apparaît dans deux splits (lecture des noms de fichiers).

Usage :
    .venv/bin/python scripts/data/split.py [--seed 0] [--ratios 0.7 0.15 0.15]
    .venv/bin/python scripts/data/split.py --check     # vérification seule
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict

from common import SPLITS, Layout, add_yolo_dir_arg, layout_from, read_manifests

# Séquences réservées au test pour la démo vidéo (make_video.py). Ignorées si la source n'est
# pas convertie. Leur groupe entier part en test.
DEMO_SEQUENCES = {
    "halmstad": [
        "V_DRONE_035",  # F450, ciel couvert, cible moyenne/petite
        "V_BIRD_020",   # oiseaux proches (~2 par frame), aucun drone : test de fausses alarmes
    ],
    "dut_track": [
        "video18",      # ciel gris, petite cible au-dessus des arbres (1080p)
        "video19",      # petite cible devant bâtiment et ciel (1080p)
    ],
}
TRAIN_ONLY = {"dut_det"}
DROPPED = "dropped"


def assign_groups(rows: list[dict], ratios: list[float], seed: int) -> dict[tuple[str, str], str]:
    """Retourne {(source, group): split} pour les sources découpées par séquence."""
    # Taille (en images) de chaque groupe, par strate (source, catégorie).
    sizes: dict[tuple[str, str], int] = defaultdict(int)
    stratum_of: dict[tuple[str, str], tuple[str, str]] = {}
    for r in rows:
        key = (r["source"], r["group"])
        sizes[key] += 1
        stratum_of[key] = (r["source"], r["category"])
    forced = {(r["source"], r["group"]) for r in rows
              if r["sequence"] in DEMO_SEQUENCES.get(r["source"], [])}

    rng = random.Random(seed)
    assignment: dict[tuple[str, str], str] = {}
    for stratum in sorted(set(stratum_of.values())):
        groups = sorted(k for k, s in stratum_of.items() if s == stratum)
        rng.shuffle(groups)
        total = sum(sizes[g] for g in groups)
        count = dict.fromkeys(SPLITS, 0)
        for g in [g for g in groups if g in forced]:
            assignment[g] = "test"
            count["test"] += sizes[g]
        for g in [g for g in groups if g not in forced]:
            # split le plus en retard sur sa cible
            deficit = {s: ratio * total - count[s] for s, ratio in zip(SPLITS, ratios)}
            split = max(SPLITS, key=lambda s: deficit[s])
            assignment[g] = split
            count[split] += sizes[g]
    return assignment


def assign_train_only(rows: list[dict], assignment: dict[tuple[str, str], str]) -> dict[tuple[str, str], str]:
    """Groupes des sources TRAIN_ONLY → train, ou écartés s'ils sont liés à un groupe val/test."""
    out = {}
    for r in rows:
        key = (r["source"], r["group"])
        if ":" in r["group"]:
            linked = tuple(r["group"].split(":", 1))
            out[key] = "train" if assignment.get(linked, "train") == "train" else DROPPED
        else:
            out[key] = "train"
    return out


def materialize(layout: Layout, rows: list[dict], assignment: dict[tuple[str, str], str]) -> None:
    """(Re)crée <yolo-dir>/<split>/{images,labels} en liens durs (copie si impossible)."""
    for split in SPLITS:
        shutil.rmtree(layout.split_dir(split), ignore_errors=True)
        for sub in ("images", "labels"):
            (layout.split_dir(split) / sub).mkdir(parents=True)
    for r in rows:
        split = assignment[(r["source"], r["group"])]
        if split == DROPPED:
            continue
        img = layout.all_images / r["image"]
        for src, sub in ((img, "images"), (layout.label_of(r["image"]), "labels")):
            dst = layout.split_dir(split) / sub / src.name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def write_split_table(layout: Layout, rows: list[dict], assignment: dict[tuple[str, str], str]) -> None:
    per_seq: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["source"], r["sequence"])
        d = per_seq.setdefault(key, {"source": r["source"], "sequence": r["sequence"], "group": r["group"],
                                     "category": r["category"], "tags": r["tags"],
                                     "split": assignment[(r["source"], r["group"])],
                                     "demo": int(r["sequence"] in DEMO_SEQUENCES.get(r["source"], [])),
                                     "n_images": 0, "n_positive": 0, "n_boxes": 0})
        d["n_images"] += 1
        d["n_positive"] += int(r["n_boxes"]) > 0
        d["n_boxes"] += int(r["n_boxes"])
    with (layout.root / "splits.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(per_seq.values())).keys()))
        w.writeheader()
        w.writerows(sorted(per_seq.values(), key=lambda d: (d["source"], d["sequence"])))


def check_leakage(layout: Layout) -> bool:
    """Vérifie, d'après les noms `<source>__<sequence>__<frame>`, qu'aucune séquence n'est partagée."""
    seen: dict[str, set[str]] = {}
    ok = True
    for split in SPLITS:
        img_dir = layout.split_dir(split) / "images"
        if not img_dir.exists():
            print(f"  {split} : absent")
            ok = False
            continue
        names = os.listdir(img_dir)
        seqs = {"__".join(n.split("__")[:2]) for n in names}
        seen[split] = seqs
        lbl_dir = layout.split_dir(split) / "labels"
        missing = sum(not (lbl_dir / (n.rsplit(".", 1)[0] + ".txt")).exists() for n in names)
        print(f"  {split:5s} : {len(names):6d} images, {len(seqs):5d} séquences, labels manquants : {missing}")
        ok &= missing == 0
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            inter = seen.get(a, set()) & seen.get(b, set())
            if inter:
                ok = False
                print(f"  FUITE {a}/{b} : {sorted(inter)[:10]}")
    print("  vérification : OK, aucune séquence partagée" if ok else "  vérification : ÉCHEC")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--sources", nargs="+", help="sources à inclure (défaut : tous les manifestes)")
    parser.add_argument("--check", action="store_true", help="vérifie seulement l'absence de fuite")
    add_yolo_dir_arg(parser)
    args = parser.parse_args()
    layout = layout_from(args.yolo_dir)

    if not args.check:
        rows = read_manifests(layout, args.sources)
        if not rows:
            raise SystemExit("aucun manifeste : lancer d'abord scripts/data/convert_*.py")
        seq_rows = [r for r in rows if r["source"] not in TRAIN_ONLY]
        assignment = assign_groups(seq_rows, args.ratios, args.seed)
        assignment.update(assign_train_only([r for r in rows if r["source"] in TRAIN_ONLY], assignment))
        materialize(layout, rows, assignment)
        write_split_table(layout, rows, assignment)
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for r in rows:
            counts[(r["source"], assignment[(r["source"], r["group"])])] += 1
        for source in sorted({r["source"] for r in rows}):
            tot = sum(counts[(source, s)] for s in (*SPLITS, DROPPED))
            line = "  ".join(f"{s} {counts[(source, s)]:6d} ({counts[(source, s)] / tot:4.0%})" for s in SPLITS)
            if counts[(source, DROPPED)]:
                line += f"  écartées {counts[(source, DROPPED)]} (liées à une séquence val/test)"
            print(f"{source:10s} {line}")
    if not check_leakage(layout):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
