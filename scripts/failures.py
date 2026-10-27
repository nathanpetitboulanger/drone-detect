"""Galerie de cas d'échec (faux négatifs / faux positifs) sur le split test.

Apparie prédictions (au seuil d'exploitation `--conf`) et vérité terrain (IoU ≥ `--iou`,
même évaluateur que `eval.py`), puis classe chaque erreur par **heuristique** (à vérifier à
l'œil : les légendes finales de `docs/RESULTS.md` sont écrites après inspection) :

Faux négatifs (drone manqué) — premier critère vérifié dans cet ordre :
- `fn_loc`        : une prédiction recouvre la cible mais mal localisée (0.1 ≤ IoU < seuil) ;
- `fn_tiny`       : cible < 16 px (sqrt(aire)) ;
- `fn_bright`     : fond très lumineux (ciel blanc, contre-jour, façade claire ; luminance
                    du voisinage ≥ 185) ;
- `fn_clutter`    : fond texturé (végétation, bâtiments : densité de contours élevée) ;
- `fn_lowcontrast`: faible contraste cible / voisinage ;
- `fn_other`.
Faux positifs (détection sans drone) :
- `fp_loc`        : recouvre un drone mais mal localisée / doublon (0.1 ≤ IoU < seuil) ;
- `fp_bird`, `fp_airplane`, `fp_helicopter` : FP dans une séquence de distracteurs dont la
                    nature est connue par son nom (Halmstad `V_BIRD_*`, `V_AIRPLANE_*`,
                    `V_HELICOPTER_*`) — pas une heuristique ;
- `fp_sky_small`  : petite détection (< 48 px) sur fond uniforme (ciel) → oiseau, avion,
                    insecte ou bruit probable ;
- `fp_bright`     : fond lumineux peu texturé → nuage, soleil, reflet probable ;
- `fp_clutter`    : fond texturé (arbres, bâtiments, sol) ;
- `fp_other`.

Pour chaque catégorie, on sauve au plus `--per-category` vignettes (au plus
`--per-seq` par séquence vidéo pour la diversité ; FP triés par score décroissant, FN tirés
au hasard avec `--seed`). Vignette = image entière réduite (zone zoomée en jaune) + zoom
sur l'erreur ; vérité en **vert**, prédictions en **rouge** (avec score). JPEG < 150 Ko.

Sorties :
- `docs/assets/failures/<categorie>__<image>__<x>_<y>.jpg` (vignettes) ;
- `docs/assets/failures/index.csv` (vignette → image, catégorie, taille, score, indices) ;
- `runs/failures/<name>/summary.json` (comptes de **toutes** les erreurs par catégorie).

Usage (depuis la racine du dépôt) :
    .venv/bin/python scripts/failures.py --weights runs/train/r1_y11n_640/weights/best.pt \
        --preds runs/eval/r1_y11n_640_640_test/predictions.json --conf 0.25 --clean
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as ek  # noqa: E402  (fixe YOLO_CONFIG_DIR avant l'import d'ultralytics)

ROOT = ek.ROOT
GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 220, 255)  # BGR

# Seuils des heuristiques (luminance 0-255 sur niveaux de gris, densité de contours Canny 0-1)
BRIGHT_CTX = 185
# Séquences de distracteurs étiquetées par leur nom (Halmstad : V_BIRD_016, V_AIRPLANE_021…)
DISTRACTOR_RE = re.compile(r"(?:^|_)(BIRD|AIRPLANE|HELICOPTER)(?:_|$)", re.IGNORECASE)
CLUTTER_EDGES = 0.08
LOW_CONTRAST = 12
SKY_EDGES = 0.03


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None, help="poids .pt (inutile si --preds)")
    ap.add_argument("--preds", default=None, help="predictions.json de eval.py (évite une 2e inférence)")
    ap.add_argument("--data", default="configs/drones.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25, help="seuil d'exploitation")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU d'appariement")
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--per-seq", type=int, default=1, help="vignettes max par séquence et par catégorie")
    ap.add_argument("--out", default="docs/assets/failures")
    ap.add_argument("--name", default=None, help="sous-dossier de runs/failures (défaut : run)")
    ap.add_argument("--clean", action="store_true", help="vide --out (jpg/csv) avant d'écrire")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# --------------------------------------------------------------------------------------
# Descripteurs d'image autour d'une boîte
# --------------------------------------------------------------------------------------
def context_window(box, w, h, factor=4.0, min_side=64):
    """Fenêtre carrée centrée sur la boîte (côté = factor × max(côté boîte), ≥ min_side)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(min_side, factor * max(box[2] - box[0], box[3] - box[1]))
    side = min(side, w, h)
    x1 = int(np.clip(cx - side / 2, 0, w - side))
    y1 = int(np.clip(cy - side / 2, 0, h - side))
    return x1, y1, int(x1 + side), int(y1 + side)


def describe(gray, edges, box) -> dict:
    """Luminance / contraste / texture autour d'une boîte."""
    h, w = gray.shape
    x1, y1, x2, y2 = context_window(box, w, h)
    ctx = gray[y1:y2, x1:x2].astype(np.float32)
    bx1, by1 = int(max(box[0], 0)), int(max(box[1], 0))
    bx2, by2 = int(min(np.ceil(box[2]), w)), int(min(np.ceil(box[3]), h))
    inside = gray[by1:by2, bx1:bx2].astype(np.float32)
    mask = np.ones_like(ctx, bool)
    mask[max(by1 - y1, 0):max(by2 - y1, 0), max(bx1 - x1, 0):max(bx2 - x1, 0)] = False
    ring = ctx[mask] if mask.any() else ctx.ravel()
    return {"img_lum": float(gray.mean()), "ctx_lum": float(ring.mean()),
            "contrast": float(abs((inside.mean() if inside.size else ring.mean()) - ring.mean())),
            "ctx_edges": float((edges[y1:y2, x1:x2] > 0)[mask].mean() if mask.any() else 0.0)}


def classify_fn(size, d, best_iou) -> str:
    if best_iou >= 0.1:
        return "fn_loc"
    if size < 16:
        return "fn_tiny"
    if d["ctx_lum"] >= BRIGHT_CTX:
        return "fn_bright"
    if d["ctx_edges"] >= CLUTTER_EDGES:
        return "fn_clutter"
    if d["contrast"] < LOW_CONTRAST:
        return "fn_lowcontrast"
    return "fn_other"


def classify_fp(size, d, best_iou, seq: str = "") -> str:
    if best_iou >= 0.1:
        return "fp_loc"
    m = DISTRACTOR_RE.search(seq)
    if m:
        return f"fp_{m.group(1).lower()}"
    if size < 48 and d["ctx_edges"] < SKY_EDGES:
        return "fp_sky_small"
    if d["ctx_lum"] >= BRIGHT_CTX - 25 and d["ctx_edges"] < CLUTTER_EDGES:
        return "fp_bright"
    if d["ctx_edges"] >= CLUTTER_EDGES:
        return "fp_clutter"
    return "fp_other"


# --------------------------------------------------------------------------------------
# Vignettes
# --------------------------------------------------------------------------------------
def draw_boxes(img, boxes, color, offset=(0, 0), scale=1.0, scores=None, thick=2):
    import cv2

    for i, b in enumerate(boxes):
        p1 = (int(round((b[0] - offset[0]) * scale)), int(round((b[1] - offset[1]) * scale)))
        p2 = (int(round((b[2] - offset[0]) * scale)), int(round((b[3] - offset[1]) * scale)))
        cv2.rectangle(img, p1, p2, color, thick)
        if scores is not None:
            cv2.putText(img, f"{scores[i]:.2f}", (p1[0], max(p1[1] - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        color, 1, cv2.LINE_AA)


def make_thumb(img, gt_boxes, pr_boxes, pr_scores, focus, title, zoom_side=300):
    """Panneau gauche : image entière réduite ; droite : zoom sur `focus`."""
    import cv2

    h, w = img.shape[:2]
    x1, y1, x2, y2 = context_window(focus, w, h, factor=6.0, min_side=96)
    crop = img[y1:y2, x1:x2]
    s = zoom_side / max(crop.shape[:2])
    interp = cv2.INTER_NEAREST if s > 2 else cv2.INTER_LINEAR
    zoom = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=interp)
    draw_boxes(zoom, gt_boxes, GREEN, (x1, y1), s)
    draw_boxes(zoom, pr_boxes, RED, (x1, y1), s, pr_scores)

    fs = zoom_side / h
    full = cv2.resize(img, (int(w * fs), zoom_side), interpolation=cv2.INTER_AREA)
    draw_boxes(full, gt_boxes, GREEN, scale=fs, thick=1)
    draw_boxes(full, pr_boxes, RED, scale=fs, thick=1)
    cv2.rectangle(full, (int(x1 * fs), int(y1 * fs)), (int(x2 * fs), int(y2 * fs)), YELLOW, 1)

    panel = np.full((zoom_side + 22, full.shape[1] + 6 + zoom.shape[1], 3), 255, np.uint8)
    panel[22:22 + full.shape[0], :full.shape[1]] = full
    panel[22:22 + zoom.shape[0], full.shape[1] + 6:full.shape[1] + 6 + zoom.shape[1]] = zoom
    cv2.putText(panel, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def save_jpeg(path: Path, img, max_bytes=150_000):
    import cv2

    for q in (80, 70, 60, 50):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and len(buf) <= max_bytes:
            break
    path.write_bytes(buf.tobytes())


# --------------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    import cv2

    data_cfg = ek.load_data_yaml(args.data)
    images = ek.list_split_images(data_cfg, args.split)
    gts = ek.load_ground_truth(images)
    if args.preds:
        preds = ek.load_preds(ek.resolve(args.preds))
        run = ek.resolve(args.preds).parent.name
    else:
        from ultralytics import YOLO

        weights = ek.resolve(args.weights)
        model = YOLO(str(weights))
        imgsz = args.imgsz or int(model.ckpt.get("train_args", {}).get("imgsz", 640))
        preds = ek.predict_yolo(model, images, imgsz, conf=args.conf, device=args.device)
        run = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    name = args.name or run

    # 1) Erreurs + descripteurs
    errors = []  # dicts
    for g in gts:
        p = ek.filter_conf(preds.get(str(g.path), ek.Pred()), args.conf)
        tp, _, hit, iou = ek.match_image(g, p, args.iou)
        fn_idx, fp_idx = np.where(~hit)[0], np.where(~tp)[0]
        if not len(fn_idx) and not len(fp_idx):
            continue
        img = cv2.imread(str(g.path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        seq = ek.sequence_of(g.path)
        for j in fn_idx:
            box = g.boxes[j]
            best = float(iou[:, j].max()) if iou.size else 0.0
            size = float(ek.box_size(box[None])[0])
            d = describe(gray, edges, box)
            errors.append({"kind": "FN", "cat": classify_fn(size, d, best), "img": g.path, "seq": seq,
                           "box": box, "size": size, "score": None, "best_iou": best, **d})
        for i in fp_idx:
            box = p.boxes[i]
            best = float(iou[i].max()) if iou.size else 0.0
            size = float(ek.box_size(box[None])[0])
            d = describe(gray, edges, box)
            errors.append({"kind": "FP", "cat": classify_fp(size, d, best, seq), "img": g.path, "seq": seq,
                           "box": box, "size": size, "score": float(p.scores[i]), "best_iou": best, **d})

    counts = Counter(e["cat"] for e in errors)
    n_gt = sum(len(g.boxes) for g in gts)
    summary = {"split": args.split, "conf": args.conf, "iou": args.iou, "n_images": len(gts), "n_gt": n_gt,
               "n_fn": sum(e["kind"] == "FN" for e in errors), "n_fp": sum(e["kind"] == "FP" for e in errors),
               "by_category": dict(sorted(counts.items())),
               "fn_size_hist": dict(Counter(ek.SIZE_LABELS[int(ek.size_bin(np.array([e["size"]]))[0])]
                                            for e in errors if e["kind"] == "FN")),
               "fp_by_sequence_top10": Counter(e["seq"] for e in errors if e["kind"] == "FP").most_common(10),
               "fn_by_sequence_top10": Counter(e["seq"] for e in errors if e["kind"] == "FN").most_common(10)}

    # 2) Échantillonnage diversifié
    rng = random.Random(args.seed)
    by_cat = defaultdict(list)
    for e in errors:
        by_cat[e["cat"]].append(e)
    chosen = []
    for cat, lst in sorted(by_cat.items()):
        if cat.startswith("fp"):
            lst = sorted(lst, key=lambda e: -e["score"])
        else:
            rng.shuffle(lst)
        per_seq = Counter()
        for e in lst:
            if per_seq[e["seq"]] >= args.per_seq:
                continue
            per_seq[e["seq"]] += 1
            chosen.append(e)
            if per_seq.total() >= args.per_category:
                break

    # 3) Vignettes
    out = ek.resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for f in list(out.glob("*.jpg")) + list(out.glob("index.csv")):
            f.unlink()
    rows = []
    for e in chosen:
        g = next(x for x in gts if x.path == e["img"])
        p = ek.filter_conf(preds.get(str(g.path), ek.Pred()), args.conf)
        img = cv2.imread(str(g.path))
        sc = f" score {e['score']:.2f}" if e["score"] is not None else ""
        title = f"{e['cat']} | {e['kind']} {e['size']:.0f}px{sc} | {e['seq'][:40]}"
        thumb = make_thumb(img, g.boxes, p.boxes, p.scores, e["box"], title)
        fname = f"{e['cat']}__{Path(e['img']).stem}__{int(e['box'][0])}_{int(e['box'][1])}.jpg"
        save_jpeg(out / fname, thumb)
        rows.append({"file": fname, "category": e["cat"], "kind": e["kind"],
                     "image": str(Path(e["img"]).relative_to(ROOT)) if Path(e["img"]).is_relative_to(ROOT)
                     else str(e["img"]), "sequence": e["seq"], "size_px": round(e["size"], 1),
                     "score": round(e["score"], 3) if e["score"] is not None else "",
                     "best_iou": round(e["best_iou"], 2), "img_lum": round(e["img_lum"]),
                     "ctx_lum": round(e["ctx_lum"]), "contrast": round(e["contrast"], 1),
                     "ctx_edges": round(e["ctx_edges"], 3)})
    if rows:
        with open(out / "index.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    summary["thumbnails"] = len(rows)
    sdir = ROOT / "runs" / "failures" / name
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"→ {len(rows)} vignettes dans {out.relative_to(ROOT)}, résumé {sdir.relative_to(ROOT)}/summary.json")


if __name__ == "__main__":
    main()
