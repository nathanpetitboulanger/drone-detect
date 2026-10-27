"""Chiffres affichés dans la vidéo, lus dans les fichiers de résultats du dépôt (aucun en dur).

Chaque valeur garde sa source (fichier + clé) ; ``python video/facts.py`` imprime le tableau
de traçabilité recopié dans ``video/STORYBOARD.md``. Si un fichier manque (ex. benchmark
TensorRT pas encore mesuré), la valeur vaut ``None`` et la scène correspondante se replie
sur une autre mesure sourcée.
"""

from __future__ import annotations

import glob
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R2_EVAL = "runs/eval/r2_y11n_1280_ft_1280_test/metrics.json"
R2_SAHI = "runs/sahi/r2_y11n_1280_ft_1280_s640at640_test/metrics.json"
R2_FAIL = "runs/failures/r2_y11n_1280_ft_1280_test/summary.json"
R1_FAIL = "runs/failures/r1_draft/summary.json"
STATS = "data/yolo/stats.md"


def _json(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def _get(d: dict, path: str):
    for k in path.split("."):
        d = d[k]
    return d


def _stats_row(label: str) -> list[str]:
    """Ligne du tableau Markdown de ``data/yolo/stats.md`` (ex. « **train** »)."""
    for line in (ROOT / STATS).read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            return cells
    raise KeyError(f"{label} absent de {STATS}")


def _bench() -> tuple[dict | None, str | None]:
    """Meilleure mesure TensorRT (FPS modèle) dans ``runs/bench/*.json``, si disponible."""
    best, src = None, None
    for f in sorted(glob.glob(str(ROOT / "runs/bench/*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows = d.get("results") or d.get("rows") or []
        for r in rows if isinstance(rows, list) else []:
            w = str(r.get("weights", ""))
            if w.endswith(".engine") and "1280" in w and r.get("fps_model"):
                if best is None or r["fps_model"] > best["fps_model"]:
                    best, src = r, str(Path(f).relative_to(ROOT))
    return best, src


@lru_cache(maxsize=1)
def facts() -> dict:
    """Dictionnaire ``nom -> (valeur, source)``."""
    out: dict[str, tuple] = {}
    ev, sahi = _json(R2_EVAL), _json(R2_SAHI)
    f2, f1 = _json(R2_FAIL), _json(R1_FAIL)

    def put(name, d, rel, path):
        out[name] = (_get(d, path), f"{rel} : {path}")

    put("map50", ev, R2_EVAL, "ultralytics_val.mAP50")
    put("map5095", ev, R2_EVAL, "ultralytics_val.mAP50-95")
    put("p50", ev, R2_EVAL, "ultralytics_val.precision")
    put("r50", ev, R2_EVAL, "ultralytics_val.recall")
    put("p30", ev, R2_EVAL, "common_eval.pr_iou03.best_f1.precision")
    put("r30", ev, R2_EVAL, "common_eval.pr_iou03.best_f1.recall")
    put("n_test_img", ev, R2_EVAL, "common_eval.pr.n_images")
    put("n_test_gt", ev, R2_EVAL, "common_eval.pr.n_gt")
    sp = ev["ultralytics_val"]["speed_ms_per_image_batched"]
    out["lat_ms_val"] = (sp["preprocess"] + sp["inference"] + sp["postprocess"],
                         f"{R2_EVAL} : ultralytics_val.speed_ms_per_image_batched (pré + inf + post)")  # fmt: skip
    put("hw_eval", ev, R2_EVAL, "hardware")
    put("dut_map50", ev, R2_EVAL, "common_eval.by_source.dut_track.coco.mAP50")  # = RESULTS « R2 @1280 »
    put("dut_sahi_map50", sahi, R2_SAHI, "modes.sahi.by_source.dut_track.coco.mAP50")
    put("dut_full_ms", sahi, R2_SAHI, "modes.full.ms_per_image_by_source.dut_track")
    put("dut_sahi_ms", sahi, R2_SAHI, "modes.sahi.ms_per_image_by_source.dut_track")
    put("r2_fp_bird", f2, R2_FAIL, "by_category.fp_bird")
    put("r1_fp_bird", f1, R1_FAIL, "by_category.fp_bird")
    # Nombre d'images d'oiseaux du test : manifeste Halmstad × splits (calculé, voir n_bird_images()).
    out["n_bird_img"] = (n_bird_images(), "data/yolo/test/images : halmstad__V_BIRD_*")

    tr, va, te, tot = (_stats_row(k) for k in ("**train**", "**val**", "**test**", "**total**"))
    out["n_train"] = (int(tr[2]), f"{STATS} : train, images")
    out["n_val"] = (int(va[2]), f"{STATS} : val, images")
    out["n_test"] = (int(te[2]), f"{STATS} : test, images")
    out["n_total"] = (int(tot[2]), f"{STATS} : total, images")
    out["n_test_seq"] = (int(te[1]), f"{STATS} : test, séquences")
    out["test_median_px"] = (float(te[8]), f"{STATS} : test, médiane √(w·h)")

    b, src = _bench()
    src = src or "runs/bench/*.json (pas encore mesuré)"
    out["trt_fps"] = (b["fps_model"] if b else None, f"{src} : fps_model (TensorRT FP16, 1280)")
    out["trt_fps_video"] = (
        (b.get("video") or {}).get("fps") if b else None,
        f"{src} : video.fps (lecture + inférence + suivi + verrou, 1080p)",
    )
    out["trt_gpu"] = (b.get("hardware") if b else None, f"{src} : hardware")
    return out


def n_bird_images() -> int:
    return len(glob.glob(str(ROOT / "data/yolo/test/images/halmstad__V_BIRD_*")))


def v(name: str):
    """Valeur seule."""
    return facts()[name][0]


def demo_defaults() -> dict:
    """Réglages du verrou **réellement utilisés** pour les clips de la vidéo.

    Lus dans les métadonnées des pistes (``outputs/video/tracks/dut_track__video19__r2.json``,
    écrites par ``track_dump.py``) ; à défaut, défauts de ``scripts/demo.py``.
    """
    p = ROOT / "outputs/video/tracks/dut_track__video19__r2.json"
    if p.exists():
        m = json.loads(p.read_text())["meta"]
        return {"lock_conf": m["lock_conf"], "min_frames": m["lock_min_frames"], "coast": m["coast_frames"],
                "tracker": m["tracker"], "conf": m["conf"], "weights": m["weights"], "source": str(p.relative_to(ROOT))}  # fmt: skip
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import demo  # noqa: E402

    a = demo.parse_args(["--source", "x"])
    return {"lock_conf": a.lock_conf, "min_frames": a.lock_min_frames, "coast": a.coast_frames,
            "tracker": a.tracker, "conf": a.conf, "weights": a.weights, "source": "scripts/demo.py (défauts)"}  # fmt: skip


def main() -> None:
    print("| Nom | Valeur | Source |\n|---|---|---|")
    for k, (val, src) in facts().items():
        print(f"| `{k}` | {val} | `{src}` |")
    print("\nRéglages démo :", demo_defaults())


if __name__ == "__main__":
    main()
