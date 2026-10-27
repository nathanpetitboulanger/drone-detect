"""Comparaison inférence par tuiles (SAHI) vs image entière, avec le même évaluateur.

Pour chaque image du split (ou d'un sous-ensemble documenté via `--stride`/`--max-images`) :
- **full** : prédiction image entière à `--imgsz` (via le wrapper SAHI → même appel
  Ultralytics que `eval.py`, mêmes seuils) ;
- **sahi** : prédiction par tuiles `--slice`×`--slice` px (recouvrement `--overlap`),
  chaque tuile inférée à `--slice-imgsz` (défaut = `--slice`, donc à pleine résolution),
  + passe image entière (`perform_standard_pred`, désactivable par `--no-standard-pred`)
  pour les grosses cibles, fusion par NMS (IoU 0.5 par défaut).

Les deux jeux de prédictions passent par l'évaluateur commun `scripts/evalkit.py`
(mAP COCO pycocotools, P/R/F1, rappel par taille de cible) et le temps par image est
mesuré hors lecture disque (image déjà décodée), avec synchronisation GPU implicite.

NMS de fusion : avec un seuil de confiance bas (< 0.1, nécessaire pour un mAP), SAHI bascule
de toute façon en NMS/IoU ; on le fixe explicitement pour éviter l'agrandissement des boîtes
de GREEDYNMM, pénalisant pour les cibles de quelques pixels.

Sorties : `runs/sahi/<name>/metrics.json`, `results_rows.md`.

Usage (depuis la racine du dépôt) :
    .venv/bin/python scripts/sahi_eval.py --weights runs/train/r1_y11n_640/weights/best.pt \
        --imgsz 640 --slice 640 --overlap 0.2
    # sous-ensemble : 1 image sur 3, au plus 1500 images
    .venv/bin/python scripts/sahi_eval.py --weights … --stride 3 --max-images 1500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as ek  # noqa: E402  (fixe YOLO_CONFIG_DIR avant l'import d'ultralytics)

ROOT = ek.ROOT


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/drones.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=None, help="taille d'inférence image entière (défaut : imgsz train)")
    ap.add_argument("--slice", type=int, default=640, help="taille des tuiles (px de l'image d'origine)")
    ap.add_argument("--slice-imgsz", type=int, default=None, help="taille d'inférence des tuiles (défaut = --slice)")
    ap.add_argument("--overlap", type=float, default=0.2, help="recouvrement des tuiles (fraction)")
    ap.add_argument("--slice-batch", type=int, default=8, help="tuiles inférées par batch")
    ap.add_argument("--no-standard-pred", action="store_true", help="tuiles seules, sans passe image entière")
    ap.add_argument("--postprocess", default="NMS", choices=["NMS", "GREEDYNMM", "NMM"])
    ap.add_argument("--match-metric", default="IOU", choices=["IOU", "IOS"])
    ap.add_argument("--match-thr", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.01, help="seuil de conf. (mAP) — identique pour les 2 modes")
    ap.add_argument("--fixed-conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=1, help="prend 1 image sur N (sous-ensemble documenté)")
    ap.add_argument("--max-images", type=int, default=0, help="limite le nb d'images (0 = toutes)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default=None, help="défaut : <run>_<imgsz>_s<slice>_<split>")
    ap.add_argument("--label", default=None)
    return ap.parse_args()


def to_pred(object_predictions) -> ek.Pred:
    """Liste d'ObjectPrediction SAHI → ek.Pred (xyxy pixels image d'origine)."""
    if not object_predictions:
        return ek.Pred()
    boxes = np.array([op.bbox.to_xyxy() for op in object_predictions], dtype=np.float64)
    scores = np.array([op.score.value for op in object_predictions], dtype=np.float64)
    cls = np.array([op.category.id for op in object_predictions], dtype=int)
    return ek.Pred(boxes, scores, cls)


def main() -> None:
    args = parse_args()
    import cv2
    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction
    from ultralytics import YOLO

    weights = ek.resolve(args.weights) if ek.resolve(args.weights).exists() else Path(args.weights)
    try:
        train_imgsz = int(YOLO(str(weights)).ckpt["train_args"]["imgsz"])
    except Exception:
        train_imgsz = 640
    imgsz = args.imgsz or train_imgsz
    slice_imgsz = args.slice_imgsz or args.slice
    run = weights.parent.parent.name if weights.parent.name == "weights" else weights.stem
    name = args.name or f"{run}_{imgsz}_s{args.slice}_{args.split}"
    out_dir = ROOT / "runs" / "sahi" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.device == "cpu" else f"cuda:{args.device}"
    data_cfg = ek.load_data_yaml(args.data)
    images = ek.list_split_images(data_cfg, args.split)[:: args.stride]
    if args.max_images:
        images = images[: args.max_images]
    gts = ek.load_ground_truth(images)
    print(f"{len(images)} images ({args.split}, stride {args.stride}), "
          f"{sum(len(g.boxes) for g in gts)} vérités terrain")

    def make_model(size: int):
        return AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                                  confidence_threshold=args.conf, device=device, image_size=size)

    full_model, slice_model = make_model(imgsz), make_model(slice_imgsz)

    preds = {"full": {}, "sahi": {}}
    times = {"full": [], "sahi": []}
    n_slices = []
    warm = cv2.cvtColor(cv2.imread(str(images[0])), cv2.COLOR_BGR2RGB)
    def sliced(img):
        return get_sliced_prediction(
            img, slice_model, slice_height=args.slice, slice_width=args.slice, overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap, perform_standard_pred=not args.no_standard_pred,
            postprocess_type=args.postprocess, postprocess_match_metric=args.match_metric,
            postprocess_match_threshold=args.match_thr, force_postprocess_type=True, verbose=0,
            batch_size=args.slice_batch)

    for _ in range(3):  # chauffe GPU / cudnn
        get_prediction(warm, full_model)
        sliced(warm)
    t_start = time.time()
    for k, img_path in enumerate(images):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        r_full = get_prediction(img, full_model)
        times["full"].append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        r_sahi = sliced(img)
        times["sahi"].append((time.perf_counter() - t0) * 1000)
        n_slices.append(_count_slices(img.shape, args.slice, args.overlap))
        preds["full"][str(img_path)] = to_pred(r_full.object_prediction_list)
        preds["sahi"][str(img_path)] = to_pred(r_sahi.object_prediction_list)
        if (k + 1) % 200 == 0:
            el = time.time() - t_start
            print(f"  {k + 1}/{len(images)} images — {el / 60:.1f} min, reste ~{el / (k + 1) * (len(images) - k - 1) / 60:.1f} min",
                  flush=True)

    hw = ek.hardware_string(args.device)
    res = {"weights": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
           "run": run, "data": args.data, "split": args.split,
           "subset": {"stride": args.stride, "max_images": args.max_images, "n_images": len(images),
                      "n_gt": int(sum(len(g.boxes) for g in gts))},
           "settings": {"imgsz_full": imgsz, "slice": args.slice, "slice_imgsz": slice_imgsz,
                        "overlap": args.overlap, "slice_batch": args.slice_batch,
                        "standard_pred": not args.no_standard_pred, "postprocess": args.postprocess,
                        "match_metric": args.match_metric, "match_thr": args.match_thr, "conf": args.conf,
                        "nms_iou": 0.7, "mean_slices_per_image": round(float(np.mean(n_slices)), 2)},
           "hardware": hw, "date": datetime.now().isoformat(timespec="seconds"), "modes": {}}
    label = args.label or run
    rows = []
    for mode in ("full", "sahi"):
        coco = ek.coco_metrics(gts, preds[mode])
        pr = ek.pr_metrics(gts, preds[mode], iou_thr=0.5, fixed_conf=args.fixed_conf)
        ms = float(np.mean(times[mode]))
        srcs = np.array([ek.source_of(g.path) for g in gts])
        t = np.array(times[mode])
        res["modes"][mode] = {"coco": coco, "pr": pr, "by_source": ek.by_source(gts, preds[mode], args.fixed_conf),
                              "ms_per_image_by_source": {s_: round(float(t[srcs == s_].mean()), 2)
                                                         for s_ in sorted(set(srcs))},
                              "ms_per_image": round(ms, 2),
                              "ms_per_image_p50": round(float(np.median(times[mode])), 2)}
        b = pr["best_f1"]
        sahi_txt = "non" if mode == "full" else f"oui ({args.slice}, ov {args.overlap})"
        rows.append(f"| {label} | {imgsz if mode == 'full' else slice_imgsz} | {sahi_txt} | {coco['mAP50']:.3f} | "
                    f"{coco['mAP50-95']:.3f} | {b['precision']:.3f} | {b['recall']:.3f} | {ms:.1f} | "
                    f"{hw.split(' (')[0]} |")
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    md = ("| Modèle | imgsz | SAHI | mAP50 | mAP50-95 | P | R | ms/image | Matériel |\n"
          "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
          f"Évaluateur commun (pycocotools, P/R au seuil F1-optimal, IoU 0.5) sur {len(images)} images "
          f"({args.split}, 1 image sur {args.stride}). Rappel par taille <16/16-32/32-96/>96 px :\n")
    for mode in ("full", "sahi"):
        rs = res["modes"][mode]["pr"]["best_f1"]["recall_by_size"]
        md += f"- {mode} : " + " | ".join(
            f"{rs[k]['recall']:.2f} ({rs[k]['n_gt']})" if rs[k]["recall"] is not None else "–"
            for k in ek.SIZE_LABELS) + "\n"
    for mode in ("full", "sahi"):
        for src, v in res["modes"][mode]["by_source"].items():
            b = v["pr"]["best_f1"]
            md += (f"- {mode} / {src} ({v['n_images']} images) : mAP50 {v['coco']['mAP50']:.3f}, "
                   f"mAP50-95 {v['coco']['mAP50-95']:.3f}, P {b['precision']:.3f}, R {b['recall']:.3f}, "
                   f"{res['modes'][mode]['ms_per_image_by_source'][src]:.1f} ms/image\n")
    (out_dir / "results_rows.md").write_text(md)
    print(json.dumps(res, indent=2))
    print("\n" + md)
    print(f"→ {out_dir.relative_to(ROOT)}/metrics.json")


def _count_slices(shape, size: int, overlap: float) -> int:
    """Nombre de tuiles SAHI pour une image (même découpage que sahi.slicing)."""
    from sahi.slicing import get_slice_bboxes

    h, w = shape[:2]
    return len(get_slice_bboxes(h, w, size, size, overlap_height_ratio=overlap, overlap_width_ratio=overlap))


if __name__ == "__main__":
    main()
