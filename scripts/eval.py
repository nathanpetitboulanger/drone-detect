"""Évaluation d'un modèle YOLO sur un split (test par défaut) du dataset drones.

Trois mesures complémentaires, sorties dans `runs/eval/<name>/` :
1. **Ultralytics** `model.val(split=…)` : mAP50, mAP50-95, P, R (P/R au seuil qui maximise
   F1, convention Ultralytics) + courbes PR/F1 (`BoxPR_curve.png`…).
2. **Évaluateur commun** (`scripts/evalkit.py`, identique à `sahi_eval.py`) sur une passe de
   prédiction image entière (conf 0.001) : mAP COCO via pycocotools, P/R/F1 au seuil
   F1-optimal et au seuil fixe `--fixed-conf`, **rappel par taille de cible**
   (< 16, 16–32, 32–96, > 96 px de sqrt(aire) dans l'image d'origine) et AP par taille,
   global puis **par source** (préfixe `<source>__` du nom de fichier).
   Les prédictions sont sauvées (`predictions.json`) pour `failures.py --preds`.
3. **Latence** batch=1 (cas temps réel) sur `--latency-n` images : pré-traitement,
   inférence, post-traitement (ms) et temps mur avec lecture de l'image.

Sorties : `metrics.json` (tout), `results_row.md` (ligne de tableau Markdown pour
`docs/RESULTS.md`), plots Ultralytics.

Usage (depuis la racine du dépôt) :
    .venv/bin/python scripts/eval.py --weights runs/train/r1_y11n_640/weights/best.pt --imgsz 640
    .venv/bin/python scripts/eval.py --weights yolo11n.pt --data coco8.yaml --split val --name chain_coco8
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evalkit as ek  # noqa: E402  (fixe YOLO_CONFIG_DIR avant l'import d'ultralytics)

ROOT = ek.ROOT


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="poids .pt (ex. runs/train/<run>/weights/best.pt)")
    ap.add_argument("--data", default="configs/drones.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=None, help="défaut : imgsz d'entraînement du modèle")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true", help="inférence FP16 (défaut FP32)")
    ap.add_argument("--conf", type=float, default=0.001, help="seuil de conf. pour le calcul des mAP")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU de la NMS")
    ap.add_argument("--fixed-conf", type=float, default=0.25, help="seuil d'exploitation pour P/R fixes")
    ap.add_argument("--latency-n", type=int, default=200, help="nb d'images pour la mesure de latence (0 = non)")
    ap.add_argument("--name", default=None, help="défaut : <run>_<imgsz>_<split>")
    ap.add_argument("--label", default=None, help="libellé du modèle dans le tableau (défaut : nom du run)")
    return ap.parse_args()


def default_imgsz(model) -> int:
    try:
        return int(model.ckpt["train_args"]["imgsz"])
    except Exception:
        return 640


def run_name(weights: Path) -> str:
    """runs/train/<run>/weights/best.pt → <run> ; yolo11n.pt → yolo11n."""
    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    weights = ek.resolve(args.weights) if ek.resolve(args.weights).exists() else Path(args.weights)
    model = YOLO(str(weights))
    imgsz = args.imgsz or default_imgsz(model)
    run = run_name(weights)
    name = args.name or f"{run}_{imgsz}_{args.split}"
    out_dir = ROOT / "runs" / "eval" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = ek.load_data_yaml(args.data)
    data_arg = str(ek.resolve(args.data)) if ek.resolve(args.data).exists() else args.data

    # 1) Ultralytics val
    m = model.val(data=data_arg, split=args.split, imgsz=imgsz, batch=args.batch, device=args.device,
                  conf=args.conf, iou=args.iou, quantize=16 if args.half else None, plots=True,
                  project=str(ROOT / "runs" / "eval"), name=name, exist_ok=True, verbose=False)
    ultra = {"mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
             "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
             "speed_ms_per_image_batched": {k: round(float(v), 3) for k, v in m.speed.items()}}

    # 2) Évaluateur commun (même code que sahi_eval.py)
    images = ek.list_split_images(data_cfg, args.split)
    gts = ek.load_ground_truth(images)
    preds = ek.predict_yolo(model, images, imgsz, conf=args.conf, iou=args.iou, device=args.device,
                            batch=args.batch, half=args.half)
    ek.save_preds(preds, out_dir / "predictions.json")
    coco = ek.coco_metrics(gts, preds)
    pr = ek.pr_metrics(gts, preds, iou_thr=0.5, fixed_conf=args.fixed_conf)
    per_source = ek.by_source(gts, preds, fixed_conf=args.fixed_conf)
    # IoU 0.3 : sur des cibles de 15–40 px, 2–3 px d'écart de boîte (conventions d'annotation
    # différentes selon les sources) font passer l'IoU sous 0.5 alors que la cible est trouvée.
    pr_iou03 = ek.pr_metrics(gts, preds, iou_thr=0.3, fixed_conf=args.fixed_conf)

    # 3) Latence batch=1
    latency = ek.measure_latency(model, images, imgsz, device=args.device, half=args.half,
                                 n=args.latency_n) if args.latency_n > 0 else {}

    hw = ek.hardware_string(args.device)
    res = {"weights": str(weights.relative_to(ROOT)) if weights.is_relative_to(ROOT) else str(weights),
           "run": run, "data": args.data, "split": args.split, "imgsz": imgsz,
           "precision_mode": "fp16" if args.half else "fp32", "hardware": hw,
           "date": datetime.now().isoformat(timespec="seconds"),
           "ultralytics_val": ultra, "common_eval": {"coco": coco, "pr": pr, "pr_iou03": pr_iou03, "by_source": per_source}, "latency_bs1": latency}
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))

    label = args.label or run
    ms = latency.get("total_ms", ultra["speed_ms_per_image_batched"].get("inference"))
    rs = pr["best_f1"]["recall_by_size"]
    row = (f"| {label} | {imgsz} | non | {ultra['mAP50']:.3f} | {ultra['mAP50-95']:.3f} | "
           f"{ultra['precision']:.3f} | {ultra['recall']:.3f} | {ms:.1f} | {hw.split(' (')[0]} |")
    size_row = " | ".join(f"{rs[k]['recall']:.2f} ({rs[k]['n_gt']})" if rs[k]["recall"] is not None else "–"
                          for k in ek.SIZE_LABELS)
    md = ("| Modèle | imgsz | SAHI | mAP50 | mAP50-95 | P | R | ms/image (bs1) | Matériel |\n"
          "|---|---|---|---|---|---|---|---|---|\n" + row + "\n\n"
          f"Rappel par taille (conf F1-opt = {pr['best_f1']['conf']:.3f}, IoU 0.5) — "
          + " / ".join(ek.SIZE_LABELS) + " px : " + size_row + "\n")
    b3 = pr_iou03["best_f1"]
    md += (f"\nÀ IoU 0.3 (localisation tolérante) : P {b3['precision']:.3f}, R {b3['recall']:.3f}, "
           f"F1 {b3['f1']:.3f} (conf {b3['conf']:.3f})\n")
    for src, v in per_source.items():
        b = v["pr"]["best_f1"]
        md += (f"\n- {src} ({v['n_images']} images, {v['pr']['n_gt']} drones) : mAP50 {v['coco']['mAP50']:.3f}, "
               f"mAP50-95 {v['coco']['mAP50-95']:.3f}, P {b['precision']:.3f}, R {b['recall']:.3f} "
               f"(conf {b['conf']:.3f}, pycocotools)")
    (out_dir / "results_row.md").write_text(md + "\n")

    print(json.dumps(res, indent=2))
    print("\n" + md)
    print(f"→ {out_dir.relative_to(ROOT)}/metrics.json")


if __name__ == "__main__":
    main()
