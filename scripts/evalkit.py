"""Outils d'évaluation partagés par `eval.py`, `sahi_eval.py` et `failures.py`.

Ce module n'est pas un script : il regroupe ce qui doit être *identique* entre les
évaluations pour que les chiffres soient comparables :

- lecture d'un split d'un dataset YOLO (yaml Ultralytics) → vérité terrain en pixels ;
- prédictions YOLO image entière (format commun : boîtes xyxy en pixels de l'image
  d'origine, score, classe) ;
- appariement prédictions / vérité par IoU (glouton, par score décroissant, par classe) ;
- évaluateur unique : mAP50 / mAP50-95 COCO via `pycocotools` + précision / rappel à un
  seuil de confiance (seuil qui maximise F1, et seuil fixe) + rappel par taille de cible.

Taille d'une cible = sqrt(w * h) en pixels de l'image d'origine (même convention que les
seuils d'aire COCO 32² / 96²). Tranches : < 16, 16–32, 32–96, > 96 px.

Les chemins relatifs sont interprétés depuis la racine du dépôt.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Réglages Ultralytics locaux au projet (datasets/, runs/) : à fixer AVANT l'import d'ultralytics.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

IMG_EXTS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic"}
SIZE_BINS = (16, 32, 96)  # bornes des tranches de taille (px, sqrt(aire))
SIZE_LABELS = ("<16", "16-32", "32-96", ">96")


# --------------------------------------------------------------------------------------
# Chemins / dataset
# --------------------------------------------------------------------------------------
def resolve(p: str | Path) -> Path:
    """Chemin relatif → relatif à la racine du dépôt (sauf s'il existe tel quel)."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (ROOT / p).resolve()


def load_data_yaml(data: str) -> dict:
    """Charge et vérifie un yaml de dataset Ultralytics (télécharge coco8 & co si besoin)."""
    from ultralytics.data.utils import check_det_dataset

    cand = resolve(data)
    return check_det_dataset(str(cand) if cand.exists() else data)


def list_split_images(data_cfg: dict, split: str) -> list[Path]:
    """Liste triée des images d'un split (dossier, fichier .txt ou liste)."""
    src = data_cfg.get(split)
    if not src:
        raise ValueError(f"split '{split}' absent du yaml de données (clés : {list(data_cfg)})")
    files: list[Path] = []
    for p in src if isinstance(src, list) else [src]:
        p = Path(p)
        if p.is_dir():
            files += [f for f in p.rglob("*.*") if f.suffix[1:].lower() in IMG_EXTS]
        elif p.is_file():
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                f = Path(line)
                if line.startswith("./") or not f.is_absolute():
                    f = p.parent / line
                files.append(f)
        else:
            raise FileNotFoundError(p)
    return sorted(files)


def label_path(img: Path) -> Path:
    """Chemin du fichier label YOLO associé à une image (…/images/… → …/labels/…)."""
    from ultralytics.data.utils import img2label_paths

    return Path(img2label_paths([str(img)])[0])


def sequence_of(img: Path) -> str:
    """Identifiant de séquence : `<source>__<sequence>` d'après la convention de nommage
    `<source>__<sequence>__<frame>.jpg` ; à défaut, le nom du dossier parent."""
    parts = img.stem.split("__")
    return "__".join(parts[:2]) if len(parts) >= 3 else img.parent.name


def source_of(img: Path) -> str:
    """Source de l'image (`halmstad`, `dut_track`…) d'après `<source>__<sequence>__<frame>.jpg`."""
    parts = img.stem.split("__")
    return parts[0] if len(parts) >= 3 else "all"


@dataclass
class GTImage:
    """Vérité terrain d'une image (boîtes xyxy en pixels de l'image d'origine)."""

    path: Path
    width: int
    height: int
    boxes: np.ndarray  # (N, 4) xyxy
    cls: np.ndarray  # (N,) int


def load_ground_truth(images: list[Path]) -> list[GTImage]:
    """Lit les labels YOLO (`cls cx cy w h` normalisés) et la taille de chaque image."""
    from PIL import Image

    out = []
    for img in images:
        with Image.open(img) as im:
            w, h = im.size  # lit seulement l'en-tête
        lp = label_path(img)
        rows = np.zeros((0, 5), dtype=np.float64)
        if lp.exists():
            txt = lp.read_text().strip()
            if txt:
                rows = np.array([list(map(float, ln.split()[:5])) for ln in txt.splitlines()], dtype=np.float64)
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        out.append(GTImage(img, w, h, boxes.reshape(-1, 4), rows[:, 0].astype(int)))
    return out


# --------------------------------------------------------------------------------------
# Prédictions
# --------------------------------------------------------------------------------------
@dataclass
class Pred:
    """Prédictions d'une image (xyxy pixels de l'image d'origine)."""

    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cls: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def predict_yolo(model, images: list[Path], imgsz: int, conf: float = 0.001, iou: float = 0.7,
                 device=None, batch: int = 16, half: bool = False, max_det: int = 300) -> dict[str, Pred]:
    """Prédictions image entière avec Ultralytics, indexées par chemin d'image (str).

    Attention : Ultralytics charge une *liste* de sources en un seul batch ; on envoie donc
    des listes de `batch` chemins et on associe les résultats par ordre."""
    preds: dict[str, Pred] = {}
    for i in range(0, len(images), batch):
        chunk = [str(p) for p in images[i:i + batch]]
        results = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, device=device,
                                quantize=16 if half else None, max_det=max_det, verbose=False)
        for path, r in zip(chunk, results):
            b = r.boxes
            preds[path] = Pred(b.xyxy.cpu().numpy().astype(np.float64), b.conf.cpu().numpy().astype(np.float64),
                               b.cls.cpu().numpy().astype(int))
    return preds


def save_preds(preds: dict[str, Pred], path: Path) -> None:
    """Sauvegarde des prédictions (JSON) pour réutilisation (ex. `failures.py --preds`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {k: {"boxes": np.round(v.boxes, 2).tolist(), "scores": np.round(v.scores, 5).tolist(),
               "cls": v.cls.tolist()} for k, v in preds.items()}
    path.write_text(json.dumps(obj))


def load_preds(path: Path) -> dict[str, Pred]:
    obj = json.loads(Path(path).read_text())
    return {k: Pred(np.array(v["boxes"], dtype=np.float64).reshape(-1, 4), np.array(v["scores"], dtype=np.float64),
                    np.array(v["cls"], dtype=int)) for k, v in obj.items()}


def filter_conf(p: Pred, conf: float) -> Pred:
    k = p.scores >= conf
    return Pred(p.boxes[k], p.scores[k], p.cls[k])


def measure_latency(model, images: list[Path], imgsz: int, device=None, half: bool = False,
                    n: int = 200, warmup: int = 10) -> dict:
    """Latence batch=1 (cas temps réel) : moyenne des temps Ultralytics pré/inf/post (ms) et
    temps mur bout en bout par image (lecture disque incluse). La charge CPU est enregistrée :
    pour un modèle nano, la latence dépend beaucoup du CPU (lancement des noyaux, pré/post)."""
    import cv2

    sel = images[:: max(1, len(images) // n)][:n] if images else []
    if not sel:
        return {}
    kw = dict(imgsz=imgsz, conf=0.25, device=device, quantize=16 if half else None, verbose=False)
    warm = cv2.imread(str(sel[0]))
    for _ in range(warmup):
        model.predict(warm, **kw)
    if _cuda_ok():
        import torch

        torch.cuda.synchronize()
    speeds, walls = [], []
    for p in sel:
        t0 = time.perf_counter()
        r = model.predict(cv2.imread(str(p)), **kw)[0]
        walls.append((time.perf_counter() - t0) * 1000)
        speeds.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
    s = np.array(speeds)
    return {"n_images": len(sel), "batch": 1, "preprocess_ms": round(float(s[:, 0].mean()), 2),
            "inference_ms": round(float(s[:, 1].mean()), 2), "postprocess_ms": round(float(s[:, 2].mean()), 2),
            "total_ms": round(float(s.sum(1).mean()), 2), "wall_ms_with_imread": round(float(np.mean(walls)), 2),
            "fps_total": round(1000 / float(s.sum(1).mean()), 1),
            "cpu_load_avg_1min": round(os.getloadavg()[0], 1), "cpu_count": os.cpu_count()}


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def hardware_string(device=None) -> str:
    """Description courte du matériel utilisé (GPU ou CPU)."""
    import platform

    import torch

    if str(device) != "cpu" and torch.cuda.is_available():
        idx = 0 if device in (None, "", "cuda") else int(str(device).split(":")[-1].split(",")[0])
        return f"{torch.cuda.get_device_name(idx)} (CUDA {torch.version.cuda}, torch {torch.__version__})"
    return f"CPU {platform.processor() or platform.machine()} (torch {torch.__version__})"


# --------------------------------------------------------------------------------------
# Appariement / métriques
# --------------------------------------------------------------------------------------
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU (N, M) entre boîtes xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).clip(0).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).clip(0).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_image(gt: GTImage, pred: Pred, iou_thr: float = 0.5):
    """Appariement glouton par score décroissant (une vérité = au plus une prédiction,
    même classe). Comme le glouton ne dépend que de l'ordre des scores, l'appariement
    calculé à conf très basse reste valable pour tout seuil plus haut.

    Retourne (pred_tp bool (P,), pred_gt idx (P,) -1 si FP, gt_hit bool (G,), iou (P, G))."""
    iou = box_iou(pred.boxes, gt.boxes)
    if iou.size:
        iou = iou * (pred.cls[:, None] == gt.cls[None, :])
    order = np.argsort(-pred.scores, kind="stable")
    tp = np.zeros(len(pred.scores), bool)
    pg = -np.ones(len(pred.scores), int)
    hit = np.zeros(len(gt.boxes), bool)
    for i in order:
        if not len(gt.boxes):
            break
        cand = np.where(~hit, iou[i], 0.0)
        j = int(cand.argmax())
        if cand[j] >= iou_thr:
            tp[i], pg[i], hit[j] = True, j, True
    return tp, pg, hit, iou


def box_size(boxes: np.ndarray) -> np.ndarray:
    """Taille d'une boîte = sqrt(w * h) en pixels."""
    wh = (boxes[:, 2:] - boxes[:, :2]).clip(0)
    return np.sqrt(wh[:, 0] * wh[:, 1])


def size_bin(s: np.ndarray) -> np.ndarray:
    return np.digitize(s, SIZE_BINS)


def pr_metrics(gts: list[GTImage], preds: dict[str, Pred], iou_thr: float = 0.5,
               fixed_conf: float = 0.25) -> dict:
    """Précision / rappel à IoU `iou_thr` : au seuil qui maximise F1 et à `fixed_conf`,
    plus le rappel par tranche de taille de cible à ces deux seuils."""
    all_scores, all_tp, n_gt = [], [], 0
    gt_best_score, gt_sizes = [], []  # score de la prédiction appariée à chaque vérité (0 si aucune)
    for g in gts:
        p = preds.get(str(g.path), Pred())
        tp, pg, hit, _ = match_image(g, p, iou_thr)
        all_scores.append(p.scores)
        all_tp.append(tp)
        n_gt += len(g.boxes)
        best = np.zeros(len(g.boxes))
        for i in np.where(tp)[0]:
            best[pg[i]] = p.scores[i]
        gt_best_score.append(best)
        gt_sizes.append(box_size(g.boxes))
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    tps = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    gbs = np.concatenate(gt_best_score) if gt_best_score else np.zeros(0)
    gsz = np.concatenate(gt_sizes) if gt_sizes else np.zeros(0)
    order = np.argsort(-scores, kind="stable")
    scores, tps = scores[order], tps[order]
    ctp, cfp = np.cumsum(tps), np.cumsum(~tps)
    prec = ctp / np.maximum(ctp + cfp, 1)
    rec = ctp / max(n_gt, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)

    def at(conf):
        k = int((scores >= conf).sum())
        if k == 0:
            return {"conf": round(float(conf), 4), "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0,
                    "fn": n_gt}
        tp_, fp_ = int(ctp[k - 1]), int(cfp[k - 1])
        return {"conf": round(float(conf), 4), "precision": round(float(prec[k - 1]), 4),
                "recall": round(float(rec[k - 1]), 4), "f1": round(float(f1[k - 1]), 4), "tp": tp_, "fp": fp_,
                "fn": n_gt - tp_}

    best_conf = float(scores[int(f1.argmax())]) if len(scores) else fixed_conf
    res = {"iou_thr": iou_thr, "n_gt": n_gt, "n_images": len(gts),
           "n_images_without_gt": int(sum(len(g.boxes) == 0 for g in gts)),
           "best_f1": at(best_conf), "fixed": at(fixed_conf)}
    for key, conf in (("best_f1", best_conf), ("fixed", fixed_conf)):
        bins = size_bin(gsz)
        res[key]["recall_by_size"] = {
            lab: {"n_gt": int((bins == b).sum()),
                  "recall": round(float((gbs[bins == b] >= conf).mean()), 4) if (bins == b).any() else None}
            for b, lab in enumerate(SIZE_LABELS)}
    return res


def coco_metrics(gts: list[GTImage], preds: dict[str, Pred]) -> dict:
    """mAP COCO (pycocotools) : AP@[.5:.95], AP50, AP75, AR@100 et AP / AR par tranche de
    taille personnalisée (< 16, 16–32, 32–96, > 96 px de sqrt(aire))."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets = [], [], []
    cats = sorted({int(c) for g in gts for c in g.cls} | {int(c) for p in preds.values() for c in p.cls})
    for i, g in enumerate(gts, 1):
        images.append({"id": i, "width": g.width, "height": g.height, "file_name": str(g.path)})
        for b, c in zip(g.boxes, g.cls):
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(c) + 1,
                         "bbox": [float(b[0]), float(b[1]), w, h], "area": w * h, "iscrowd": 0})
        p = preds.get(str(g.path), Pred())
        for b, s, c in zip(p.boxes, p.scores, p.cls):
            dets.append({"image_id": i, "category_id": int(c) + 1,
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                         "score": float(s)})
    keys = ["mAP50-95", "mAP50", "mAP75", "AR100"] + [f"AP_{s}" for s in SIZE_LABELS] + [f"AR_{s}" for s in
                                                                                          SIZE_LABELS]
    if not anns or not dets:
        return {k: 0.0 for k in keys}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": c + 1} for c in cats]}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        b = [0] + [x ** 2 for x in SIZE_BINS] + [1e10]
        ev.params.areaRng = [[0, 1e10]] + [[b[i], b[i + 1]] for i in range(len(b) - 1)]
        ev.params.areaRngLbl = ["all", *SIZE_LABELS]
        ev.evaluate()
        ev.accumulate()

    def ap(iou=None, area="all"):
        a = ev.params.areaRngLbl.index(area)
        m = ev.params.maxDets.index(100)
        s = ev.eval["precision"]  # [T, R, K, A, M]
        if iou is not None:
            s = s[np.where(np.isclose(ev.params.iouThrs, iou))[0]]
        s = s[:, :, :, a, m]
        return float(np.mean(s[s > -1])) if (s > -1).any() else None

    def ar(area="all"):
        a = ev.params.areaRngLbl.index(area)
        s = ev.eval["recall"][:, :, a, ev.params.maxDets.index(100)]  # [T, K, A, M]
        return float(np.mean(s[s > -1])) if (s > -1).any() else None

    out = {"mAP50-95": ap(), "mAP50": ap(0.5), "mAP75": ap(0.75), "AR100": ar()}
    for s in SIZE_LABELS:
        out[f"AP_{s}"], out[f"AR_{s}"] = ap(area=s), ar(area=s)
    return {k: (round(v, 4) if v is not None else None) for k, v in out.items()}


def by_source(gts: list[GTImage], preds: dict[str, Pred], fixed_conf: float = 0.25) -> dict:
    """Mêmes métriques (mAP COCO + P/R/F1 + rappel par taille) restreintes à chaque source :
    les sources n'ont ni la même résolution ni la même difficulté (cf. docs/DATA.md)."""
    groups: dict[str, list[GTImage]] = {}
    for g in gts:
        groups.setdefault(source_of(g.path), []).append(g)
    if len(groups) < 2:
        return {}
    return {src: {"n_images": len(sub), "coco": coco_metrics(sub, preds),
                  "pr": pr_metrics(sub, preds, iou_thr=0.5, fixed_conf=fixed_conf)}
            for src, sub in sorted(groups.items())}
