"""Fine-tuning d'un YOLO Ultralytics pré-entraîné COCO pour détecter des drones.

Wrapper argparse autour de `YOLO(model).train(...)` qui :
- range les runs dans `runs/train/<name>/` (chemin absolu → pas de sous-dossier
  `runs/detect/…` ajouté par Ultralytics) ;
- enregistre la commande exacte, les hyperparamètres, les versions et le matériel dans
  `runs/train/<name>/run_config.json` dès le début du run, puis un résumé
  (`run_summary.json` : meilleure époque, métriques val, durée) à la fin ;
- applique un preset d'augmentations adapté aux petites cibles (voir `AUG_PRESETS`).

Augmentations (preset `small`, par défaut) — on garde les défauts Ultralytics sauf raison :
- `mosaic=1.0` (défaut) : multiplie les contextes et les cibles par image, utile pour les
  petits objets ; coupée sur les `close_mosaic=10` dernières époques (défaut) pour finir sur
  des images « naturelles ».
- `scale=0.3` (défaut 0.5) : à 0.5 le zoom arrière peut diviser la taille par 2 et rendre
  une cible de 10 px quasi invisible (5 px, sous la résolution utile du stride 8). 0.3 garde
  de la variation de distance sans détruire les cibles minuscules.
- `fliplr=0.5` (défaut), `flipud=0.0` (défaut : le ciel est en haut, pas d'inversion).
- `degrees=0`, `shear=0`, `perspective=0` (défauts) : les rotations gonflent les boîtes
  axis-alignées, très pénalisant à l'IoU pour des cibles de quelques pixels.
- `hsv_h/s/v` (défauts 0.015/0.7/0.4) : variations de lumière (contre-jour, ciel gris).
- `mixup=0`, `cutmix=0` (défauts) : peu utiles en mono-classe, brouillent les petites cibles.
Preset `default` = défauts Ultralytics purs. Tout hyperparamètre Ultralytics peut aussi être
passé via `--set clé=valeur` (répétable).

Usage (depuis la racine du dépôt) :
    # validation de chaîne (tiny dataset COCO8, téléchargé automatiquement, ~1 Mo)
    .venv/bin/python scripts/train.py --data coco8.yaml --epochs 2 --name chain_coco8 --exist-ok

    # run court sur les données drones
    .venv/bin/python scripts/train.py --data configs/drones.yaml --epochs 5 --fraction 0.2 --name short_v0

    # vrai run (survit à la fermeture du terminal)
    mkdir -p runs/logs && nohup env YOLO_CONFIG_DIR=$PWD/.ultralytics .venv/bin/python scripts/train.py \
        --data configs/drones.yaml --model yolo11n.pt --imgsz 640 --epochs 100 --patience 20 \
        --time 3.5 --name r1_y11n_640 > runs/logs/r1_y11n_640.log 2>&1 &

    # reprendre un run interrompu
    .venv/bin/python scripts/train.py --resume runs/train/r1_y11n_640/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))  # avant l'import d'ultralytics

AUG_PRESETS = {
    "small": dict(mosaic=1.0, close_mosaic=10, scale=0.3, translate=0.1, fliplr=0.5, flipud=0.0, degrees=0.0,
                  shear=0.0, perspective=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mixup=0.0, cutmix=0.0),
    "default": {},
}


def parse_value(v: str):
    """'0.5' → 0.5, 'true' → True, 'none' → None, sinon chaîne."""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n.pt", help="poids pré-entraînés COCO (yolo11n.pt, yolo11s.pt, yolov8s.pt…)")
    ap.add_argument("--data", default="configs/drones.yaml", help="yaml dataset Ultralytics")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--time", type=float, default=None, help="durée max en heures (prioritaire sur --epochs)")
    ap.add_argument("--patience", type=int, default=20, help="early stopping (époques sans amélioration)")
    ap.add_argument("--batch", type=float, default=16, help="taille de batch (-1 = AutoBatch, 0.x = fraction VRAM)")
    ap.add_argument("--name", default=None, help="nom du run (défaut : <modèle>_<imgsz>)")
    ap.add_argument("--project", default="runs/train", help="dossier des runs (relatif à la racine du dépôt)")
    ap.add_argument("--exist-ok", action="store_true", help="écrase runs/train/<name> au lieu d'incrémenter")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction du train utilisée (1.0 = tout)")
    ap.add_argument("--cache", default="false", choices=["false", "ram", "disk"], help="cache des images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0", help="'0' (GPU), 'cpu'…")
    ap.add_argument("--optimizer", default="auto")
    ap.add_argument("--lr0", type=float, default=None, help="LR initial (défaut Ultralytics selon optimiseur)")
    ap.add_argument("--cos-lr", action="store_true", help="scheduler cosinus")
    ap.add_argument("--close-mosaic", type=int, default=None, help="désactive la mosaïque sur les N dernières époques")
    ap.add_argument("--aug", default="small", choices=list(AUG_PRESETS), help="preset d'augmentations")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="hyperparamètre Ultralytics supplémentaire (répétable), ex. --set scale=0.5")
    ap.add_argument("--resume", default=None, metavar="LAST_PT", help="reprend un run depuis weights/last.pt")
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args()


def env_info(device: str) -> dict:
    """Versions et matériel, pour la traçabilité."""
    import torch
    import ultralytics

    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except Exception:
        commit = ""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else None
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__, "gpu": gpu, "git_commit": commit}


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)  # chemins relatifs = racine du dépôt (poids pré-entraînés téléchargés ici aussi)
    from ultralytics import YOLO

    command = " ".join(shlex.quote(a) for a in [".venv/bin/python", "scripts/train.py", *sys.argv[1:]])
    started = time.time()

    if args.resume:
        model = YOLO(args.resume)
        kwargs = {"resume": True}
    else:
        data = Path(args.data)
        data = str(data.resolve()) if data.exists() else args.data  # sinon dataset intégré (coco8.yaml…)
        name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
        kwargs = dict(data=data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience,
                      batch=int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch,
                      project=str((ROOT / args.project).resolve()), name=name, exist_ok=args.exist_ok,
                      fraction=args.fraction, cache=False if args.cache == "false" else args.cache,
                      workers=args.workers, seed=args.seed, device=args.device, optimizer=args.optimizer,
                      cos_lr=args.cos_lr, plots=not args.no_plots)
        if args.time:
            kwargs["time"] = args.time
        if args.lr0 is not None:
            kwargs["lr0"] = args.lr0
        kwargs.update(AUG_PRESETS[args.aug])
        if args.close_mosaic is not None:
            kwargs["close_mosaic"] = args.close_mosaic
        for kv in args.set:
            k, _, v = kv.partition("=")
            kwargs[k.strip()] = parse_value(v.strip())
        model = YOLO(args.model)

    info = {"command": command, "cwd": str(ROOT), "started": datetime.now().isoformat(timespec="seconds"),
            "script_args": vars(args), "train_kwargs": kwargs, "env": env_info(str(args.device))}

    def on_pretrain_routine_start(trainer):
        """Écrit run_config.json dès que le dossier du run est connu (utile si le run plante)."""
        save_dir = Path(trainer.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "run_config.json").write_text(json.dumps(info, indent=2, default=str))

    model.add_callback("on_pretrain_routine_start", on_pretrain_routine_start)
    model.train(**kwargs)

    trainer = model.trainer
    save_dir = Path(trainer.save_dir)
    summary = {
        "save_dir": str(save_dir.relative_to(ROOT)) if save_dir.is_relative_to(ROOT) else str(save_dir),
        "best_weights": str(Path(trainer.best).relative_to(ROOT)) if Path(trainer.best).is_relative_to(ROOT)
        else str(trainer.best),
        "best_epoch": None,  # rempli ci-dessous depuis results.csv
        "epochs_run": int(trainer.epoch) + 1,
        "best_fitness": float(trainer.best_fitness) if trainer.best_fitness is not None else None,
        "final_val_metrics": {k: float(v) for k, v in (trainer.metrics or {}).items()},
        "duration_h": round((time.time() - started) / 3600, 3),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    # meilleure époque d'après results.csv (fitness Ultralytics 8.4 = mAP50-95 seul, cf. DetMetrics.fitness)
    csv = save_dir / "results.csv"
    if csv.exists():
        import pandas as pd

        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        if "metrics/mAP50-95(B)" in df.columns:
            i = int(df["metrics/mAP50-95(B)"].idxmax())
            summary["best_epoch"] = int(df.loc[i, "epoch"])
            summary["best_epoch_val"] = {k: float(df.loc[i, k]) for k in df.columns if k.startswith("metrics/")}
    (save_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
