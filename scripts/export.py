"""Export ONNX / TensorRT d'un modèle Ultralytics YOLO + vérification des sorties.

Exporte des poids ``.pt`` (YOLO11n fine-tuné drones, ou ``yolo11n.pt`` COCO pour le
développement) vers des formats déployables sur Jetson, via ``model.export(...)`` :

- **ONNX** (``--formats onnx``) : FP32, batch 1, opset ``--opset`` (17 par défaut : lu par
  TensorRT 8.5+ donc JetPack 5 et 6), graphe simplifié (onnxslim), forme fixe ou axes
  dynamiques (``--dynamic`` : batch, hauteur, largeur). C'est l'artefact portable : sur le
  Jetson, on reconstruit l'engine TensorRT **sur place** à partir de ce fichier (voir
  docs/EXPORT.md).
- **TensorRT** (``--formats engine``, GPU obligatoire) : engine ``.engine`` FP16 par défaut
  (``--precision fp32|int8`` possibles ; INT8 calibré sur ``--data``). Un engine est lié au
  GPU et à la version de TensorRT qui l'ont construit : celui-ci ne sert qu'aux mesures sur
  la machine locale.

Noms de sortie (dans ``--out``, par défaut à côté des poids) :
    <stem>_<imgsz>[_dyn].onnx            ex. best_1280.onnx
    <stem>_<imgsz>[_dyn]_<prec>.engine   ex. best_1280_fp16.engine

Vérification (sauf ``--no-check``) : prédiction PyTorch FP32 vs modèle exporté sur les
mêmes images (``--check-images`` ; par défaut bus.jpg et zidane.jpg d'Ultralytics +
``--check-n`` images réparties du split test drones s'il existe), même letterbox carrée ``imgsz`` des deux côtés,
même device. Pour chaque image : appariement glouton des boîtes (même classe, IoU), IoU
moyenne / min, écart max de confiance et de coordonnées (px). Verdict « OK » si toute
boîte de confiance >= ``--check-conf`` + 0.05 d'un côté a son homologue (IoU >= 0.9) de
l'autre côté (les boîtes proches du seuil peuvent légitimement apparaître/disparaître).

``--skip-existing`` réutilise un fichier exporté déjà présent (pas de reconstruction de
l'engine, quelques minutes) et ne refait que la vérification.

Résumé (tailles, temps d'export, vérification, versions) affiché et écrit dans
``<out>/<stem>_export.json`` (fusionné d'un appel à l'autre, une entrée par fichier).

Usage (depuis la racine du dépôt) :
    # dev sur CPU avec le modèle COCO, ONNX 640 et 1280
    .venv/bin/python scripts/export.py --weights yolo11n.pt --imgsz 640 1280 --out runs/export/yolo11n
    # ONNX + engine TensorRT FP16 (GPU)
    .venv/bin/python scripts/export.py --weights runs/train/<run>/weights/best.pt --imgsz 1280 \\
        --formats onnx engine --device 0
    # ONNX à axes dynamiques (batch, hauteur, largeur)
    .venv/bin/python scripts/export.py --weights yolo11n.pt --imgsz 640 --dynamic --out runs/export/yolo11n
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))  # avant l'import d'ultralytics
sys.path.insert(0, str(Path(__file__).resolve().parent))

PRECISIONS = {"fp32": None, "fp16": 16, "int8": 8}  # --precision → argument `quantize` d'Ultralytics
STABLE_MARGIN = 0.05  # marge au-dessus de --check-conf pour qu'une boîte soit « stable »
MATCH_IOU = 0.9  # IoU minimale entre une boîte stable et son homologue


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="poids PyTorch .pt à exporter")
    ap.add_argument(
        "--formats",
        nargs="+",
        choices=["onnx", "engine"],
        default=["onnx"],
        help="formats à produire (engine = TensorRT, GPU obligatoire) (défaut : onnx)",
    )
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640], help="taille(s) d'entrée carrée(s) (défaut : 640)")
    ap.add_argument("--out", type=Path, default=None, help="dossier de sortie (défaut : dossier des poids)")
    ap.add_argument(
        "--device",
        default=None,
        help="device d'export et de vérification : cpu, 0… (défaut : 0 si engine demandé, sinon cpu)",
    )
    ap.add_argument("--opset", type=int, default=17, help="opset ONNX (défaut : 17)")
    ap.add_argument("--no-simplify", action="store_true", help="ne pas simplifier le graphe ONNX (onnxslim)")
    ap.add_argument(
        "--dynamic", action="store_true", help="axes dynamiques (batch, hauteur, largeur) ; imgsz = taille de référence"
    )
    ap.add_argument("--nms", action="store_true", help="intégrer la NMS dans le modèle exporté (défaut : NMS externe)")
    ap.add_argument(
        "--precision", choices=list(PRECISIONS), default="fp16", help="précision de l'engine TensorRT (défaut : fp16)"
    )
    ap.add_argument("--data", default="configs/drones.yaml", help="dataset de calibration INT8 (--precision int8)")
    ap.add_argument("--workspace", type=float, default=None, help="workspace TensorRT en Gio (défaut : auto)")
    ap.add_argument("--check-images", nargs="+", default=None, help="images ou dossiers pour la vérification")
    ap.add_argument("--check-conf", type=float, default=0.25, help="seuil de confiance de la vérification")
    ap.add_argument("--check-n", type=int, default=6, help="images du split test ajoutées à la vérification")
    ap.add_argument("--no-check", action="store_true", help="ne pas vérifier les modèles exportés")
    ap.add_argument("--skip-existing", action="store_true", help="réutiliser les fichiers déjà exportés")
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="threads CPU max (torch, OpenCV, ONNX Runtime) (défaut : réglage des bibliothèques)",
    )
    return ap.parse_args(argv)  # fmt: skip


# --- Utilitaires -----------------------------------------------------------------------


def resolve(p: str | Path) -> Path:
    """Chemin relatif → relatif à la racine du dépôt (sauf s'il existe tel quel)."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (ROOT / p).resolve()


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def mb(p: Path) -> float:
    return Path(p).stat().st_size / 1e6


def versions() -> dict:
    """Versions des bibliothèques d'export / d'inférence (None si absente)."""
    import importlib

    out = {}
    for mod in ("torch", "ultralytics", "onnx", "onnxslim", "onnxruntime", "tensorrt"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except Exception:
            out[mod] = None
    return out


def device_label(device: str) -> str:
    import torch

    if device != "cpu" and torch.cuda.is_available():
        return f"GPU {torch.cuda.get_device_name(int(device.split(':')[-1]) if device[-1].isdigit() else 0)}"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return f"CPU {line.split(':', 1)[1].strip()}"
    except OSError:
        pass
    return "CPU"


def default_check_images(n_test: int) -> list[Path]:
    """bus.jpg / zidane.jpg d'Ultralytics + quelques images réparties du split test drones."""
    from ultralytics.utils import ASSETS

    imgs = sorted(Path(ASSETS).glob("*.jpg"))
    test_dir = ROOT / "data" / "yolo" / "test" / "images"
    if test_dir.is_dir():
        test = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if test:
            idx = np.linspace(0, len(test) - 1, min(n_test, len(test))).round().astype(int)
            imgs += [test[i] for i in sorted(set(idx))]
    return imgs


def collect_images(items: list[str]) -> list[Path]:
    out = []
    for it in items:
        p = resolve(it)
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"image de vérification introuvable : {it}")
    return out


# --- Vérification ----------------------------------------------------------------------


def predict_boxes(model, images: list[Path], imgsz: int, device: str, conf: float) -> list[dict]:
    """Boîtes par image (xyxy px de l'image d'origine, conf, classe), letterbox carrée imgsz."""
    out = []
    for img in images:
        r = model.predict(str(img), imgsz=imgsz, conf=conf, device=device, rect=False, verbose=False)[0]
        b = r.boxes
        out.append({
            "xyxy": b.xyxy.cpu().numpy().astype(float),
            "conf": b.conf.cpu().numpy().astype(float),
            "cls": b.cls.cpu().numpy().astype(int),
        })  # fmt: skip
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU deux à deux entre boîtes xyxy (N×4, M×4) → N×M."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def compare_boxes(ref: dict, exp: dict, conf_thr: float) -> dict:
    """Appariement glouton (même classe, IoU décroissante) entre prédictions de référence et exportées."""
    iou = box_iou(ref["xyxy"], exp["xyxy"])
    if iou.size:
        iou[ref["cls"][:, None] != exp["cls"][None, :]] = 0.0
    pairs, used_r, used_e = [], set(), set()
    for flat in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] < 0.5:
            break
        if i in used_r or j in used_e:
            continue
        used_r.add(i)
        used_e.add(j)
        pairs.append((i, j, float(iou[i, j])))

    stable = conf_thr + STABLE_MARGIN
    good_r = {i for i, _, v in pairs if v >= MATCH_IOU}
    good_e = {j for _, j, v in pairs if v >= MATCH_IOU}
    missed = sum(1 for i, c in enumerate(ref["conf"]) if c >= stable and i not in good_r)
    extra = sum(1 for j, c in enumerate(exp["conf"]) if c >= stable and j not in good_e)
    ious = [v for *_, v in pairs]
    return {
        "n_ref": len(ref["conf"]),
        "n_exp": len(exp["conf"]),
        "n_matched": len(pairs),
        "iou_mean": float(np.mean(ious)) if ious else None,
        "iou_min": float(np.min(ious)) if ious else None,
        "dconf_max": max((abs(ref["conf"][i] - exp["conf"][j]) for i, j, _ in pairs), default=None),
        "dxyxy_max_px": max((float(np.abs(ref["xyxy"][i] - exp["xyxy"][j]).max()) for i, j, _ in pairs),
                            default=None),
        "stable_missed": missed,
        "stable_extra": extra,
    }  # fmt: skip


def summarize_check(per_image: list[dict], images: list[Path]) -> dict:
    def agg(key, fn):
        vals = [d[key] for d in per_image if d[key] is not None]
        return float(fn(vals)) if vals else None

    ok = all(d["stable_missed"] == 0 and d["stable_extra"] == 0 for d in per_image)
    return {
        "verdict": "OK" if ok else "ÉCART",
        "n_images": len(per_image),
        "n_ref": sum(d["n_ref"] for d in per_image),
        "n_exp": sum(d["n_exp"] for d in per_image),
        "n_matched": sum(d["n_matched"] for d in per_image),
        "iou_mean": agg("iou_mean", np.mean),
        "iou_min": agg("iou_min", np.min),
        "dconf_max": agg("dconf_max", np.max),
        "dxyxy_max_px": agg("dxyxy_max_px", np.max),
        "stable_missed": sum(d["stable_missed"] for d in per_image),
        "stable_extra": sum(d["stable_extra"] for d in per_image),
        "per_image": {rel(p): d for p, d in zip(images, per_image, strict=True)},
    }


def fmt_check(c: dict | None) -> str:
    if not c:
        return "non vérifié"
    if c.get("error"):
        return f"ERREUR : {c['error']}"
    f = lambda v, spec: "—" if v is None else format(v, spec)  # noqa: E731
    return (f"{c['verdict']} — {c['n_matched']}/{c['n_ref']} boîtes PyTorch appariées ({c['n_exp']} exportées), "
            f"IoU moy {f(c['iou_mean'], '.4f')} min {f(c['iou_min'], '.4f')}, Δconf max {f(c['dconf_max'], '.1e')}, "
            f"Δcoord max {f(c['dxyxy_max_px'], '.1e')} px, "
            f"stables manquées/en trop {c['stable_missed']}/{c['stable_extra']}")  # fmt: skip


# --- Export ----------------------------------------------------------------------------


def output_path(weights: Path, out_dir: Path, fmt: str, imgsz: int, args) -> Path:
    base = f"{weights.stem}_{imgsz}{'_dyn' if args.dynamic else ''}"
    return out_dir / (f"{base}.onnx" if fmt == "onnx" else f"{base}_{args.precision}.engine")


def export_one(weights: Path, out_dir: Path, fmt: str, imgsz: int, args, device: str) -> tuple[Path, float]:
    """Exporte via une copie de travail des poids nommée comme la sortie voulue.

    Ultralytics écrit ``<poids>.onnx`` / ``<poids>.engine`` à côté des poids : la copie
    ``<out>/<base>.pt`` donne directement le nom final et évite d'écraser un export
    existant (l'export engine passe lui-même par un ONNX intermédiaire, supprimé ensuite).
    """
    from ultralytics import YOLO

    expected = output_path(weights, out_dir, fmt, imgsz, args)
    kwargs = dict(format=fmt, imgsz=imgsz, batch=1, opset=args.opset, simplify=not args.no_simplify,
                  dynamic=args.dynamic, device=device, verbose=False)  # fmt: skip
    if args.nms:
        kwargs["nms"] = True
    if fmt == "engine":
        if PRECISIONS[args.precision]:
            kwargs["quantize"] = PRECISIONS[args.precision]
        if args.precision == "int8":
            kwargs["data"] = str(resolve(args.data))
        if args.workspace:
            kwargs["workspace"] = args.workspace

    work = expected.with_suffix(".pt")
    if work.resolve() == weights.resolve():
        raise SystemExit(f"collision de nom avec les poids d'origine : {work}")
    shutil.copy2(weights, work)
    try:
        t0 = time.perf_counter()
        f = Path(YOLO(str(work)).export(**kwargs))
        dt = time.perf_counter() - t0
    finally:
        work.unlink(missing_ok=True)
        if fmt == "engine":
            work.with_suffix(".onnx").unlink(missing_ok=True)  # ONNX intermédiaire de l'engine
    if f.resolve() != expected.resolve():  # par sécurité si Ultralytics change ses conventions
        shutil.move(str(f), expected)
    return expected, dt


def check_onnx_graph(path: Path) -> str:
    import onnx

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    shape = lambda v: [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    ins = ", ".join(f"{i.name}{shape(i)}" for i in m.graph.input)
    outs = ", ".join(f"{o.name}{shape(o)}" for o in m.graph.output)
    return f"opset {m.opset_import[0].version}, entrée {ins}, sortie {outs}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import torch
    from bench import tune_runtime
    from ultralytics import YOLO

    tune_runtime(args.threads)
    weights = resolve(args.weights)
    if not weights.is_file() or weights.suffix != ".pt":
        raise SystemExit(f"poids .pt introuvables : {args.weights}")
    out_dir = resolve(args.out) if args.out else weights.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    want_engine = "engine" in args.formats
    device = str(args.device) if args.device is not None else ("0" if want_engine else "cpu")
    if want_engine and (device == "cpu" or not torch.cuda.is_available()):
        raise SystemExit("l'export TensorRT exige un GPU CUDA (--device 0)")
    if args.precision == "int8" and want_engine and not resolve(args.data).exists():
        raise SystemExit(f"dataset de calibration INT8 introuvable : {args.data}")

    check_images = [] if args.no_check else (
        collect_images(args.check_images) if args.check_images else default_check_images(args.check_n)
    )  # fmt: skip
    vers, hw = versions(), device_label(device)
    print(f"[export] poids {rel(weights)} ({mb(weights):.1f} Mo) → {rel(out_dir)}")
    print(f"[export] formats {args.formats}, imgsz {args.imgsz}, device {device} ({hw})")
    print("[export] versions : " + ", ".join(f"{k} {v}" for k, v in vers.items() if v))

    report_path = out_dir / f"{weights.stem}_export.json"
    previous = {}
    if report_path.exists():
        try:
            previous = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            previous = {}
    ref_model, ref_cache = None, {}
    results = []
    for imgsz in args.imgsz:
        # ONNX d'abord : il ne dépend pas du GPU et sert de base au déploiement Jetson.
        for fmt in sorted(args.formats, key=lambda x: x != "onnx"):
            print(f"\n[export] === {fmt} imgsz {imgsz}{' dynamique' if args.dynamic else ''} ===")
            entry = {
                "format": fmt, "imgsz": imgsz, "dynamic": args.dynamic, "nms": args.nms, "opset": args.opset,
                "simplify": not args.no_simplify, "precision": args.precision if fmt == "engine" else "fp32",
                "device": device, "hardware": hw, "source_weights": rel(weights),
                "date": datetime.now().isoformat(timespec="seconds"), "versions": vers,
            }  # fmt: skip
            path = output_path(weights, out_dir, fmt, imgsz, args)
            if args.skip_existing and path.exists():
                dt = previous.get(path.name, {}).get("export_s")
                print(f"[export] {rel(path)} existe déjà ({mb(path):.1f} Mo) : réutilisé (--skip-existing)")
            else:
                path, dt = export_one(weights, out_dir, fmt, imgsz, args, device)
                dt = round(dt, 1)
                print(f"[export] {rel(path)} : {mb(path):.1f} Mo en {dt:.1f} s")
            entry.update(file=rel(path), size_mb=round(mb(path), 2), export_s=dt)
            if fmt == "onnx":
                entry["graph"] = check_onnx_graph(path)
                print(f"[export] onnx.checker OK — {entry['graph']}")

            if check_images:
                try:
                    if ref_model is None:
                        ref_model = YOLO(str(weights))
                    key = (imgsz, device)
                    if key not in ref_cache:
                        ref_cache[key] = predict_boxes(ref_model, check_images, imgsz, device, args.check_conf)
                    exp_model = YOLO(str(path), task=ref_model.task)
                    exp_boxes = predict_boxes(exp_model, check_images, imgsz, device, args.check_conf)
                    per_image = [
                        compare_boxes(r, e, args.check_conf) for r, e in zip(ref_cache[key], exp_boxes, strict=True)
                    ]
                    entry["check"] = summarize_check(per_image, check_images)
                    entry["check"]["conf"] = args.check_conf
                    del exp_model
                except Exception as e:  # on garde l'export même si la vérification plante
                    entry["check"] = {"verdict": "ERREUR", "error": f"{type(e).__name__}: {e}"}
                print(f"[export] vérification vs PyTorch ({len(check_images)} images) : {fmt_check(entry['check'])}")
            results.append(entry)

    # Rapport JSON fusionné (une entrée par fichier exporté).
    report = {**previous, **{Path(e["file"]).name: e for e in results}}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float))

    print(f"\n[export] résumé ({hw}) :")
    w = max(len(rel(weights)), *(len(e["file"]) for e in results))
    print(f"  {'fichier':<{w}} {'taille':>8}  {'export':>7}  vérification")
    print(f"  {rel(weights):<{w}} {mb(weights):>6.1f} Mo  {'—':>7}  (référence PyTorch)")
    for e in results:
        verdict = e.get("check", {}).get("verdict", "—")
        t = "—" if e["export_s"] is None else f"{e['export_s']:.1f} s"
        print(f"  {e['file']:<{w}} {e['size_mb']:>6.1f} Mo  {t:>7}  {verdict}")
    print(f"[export] rapport : {rel(report_path)}")
    if any(e.get("check", {}).get("verdict") in {"ÉCART", "ERREUR"} for e in results):
        sys.exit(2)


if __name__ == "__main__":
    main()
