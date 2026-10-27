"""Enregistre, image par image, les pistes et l'état du verrou de la démo pour la vidéo.

Même boucle que ``scripts/demo.py`` (même modèle, ``model.track`` + ByteTrack, même machine
à états ``scripts/lockon.py``, mêmes réglages par défaut, lus dans ``demo.parse_args``),
mais au lieu d'incruster l'overlay on sauvegarde tout ce qu'il faut pour le **redessiner**
avec la fonction ``demo.draw_overlay`` à la résolution de la vidéo de présentation (les
clips Halmstad 640×512 sont agrandis : un overlay incrusté à 640 px serait flou). Les
mêmes données alimentent la loupe et le graphe dx/dy synchronisé.

Sortie : un JSON par (source, poids) dans ``outputs/video/tracks/`` :
    meta    poids, imgsz, tracker, réglages du verrou, matériel, taille et fps de la source,
            FPS pipeline moyen (hors 1re frame), latence modèle moyenne
    frames  liste par frame : ``i`` (index), ``fps`` (EMA, comme l'overlay de la démo),
            ``state``, ``lost``, ``new``, ``n`` (pistes), ``tgt`` [id, x1, y1, x2, y2, conf]
            ou null, ``aim`` [dx, dy, area_ratio] ou null, ``tracks`` [[id, x1, y1, x2, y2,
            conf, cls]…], ``dets`` [[x1, y1, x2, y2, conf, cls]…] (détections sans ID)

Usage (depuis la racine du dépôt ; GPU de préférence pour que les FPS affichés soient
ceux du GPU, sinon ``--device cpu``) :
    export YOLO_CONFIG_DIR=$PWD/.ultralytics
    E=runs/train/r2_y11n_1280_ft/weights/best_1280_fp16.engine          # mêmes réglages que outputs/demo/
    .venv/bin/python video/track_dump.py --device 0 --weights $E --conf 0.1 --lock-conf 0.4
    .venv/bin/python video/track_dump.py --device 0 --weights runs/train/r1_y11n_640/weights/best.pt \\
        --imgsz 640 --conf 0.1 --lock-conf 0.4 --sources data/demo/halmstad__V_BIRD_020.mp4 --tag r1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
sys.path.insert(0, str(ROOT / "scripts"))
import demo  # noqa: E402
from lockon import LockOn, LockState  # noqa: E402

R2 = "runs/train/r2_y11n_1280_ft/weights/best.pt"
DEMO_SOURCES = [
    "data/demo/dut_track__video18.mp4",
    "data/demo/dut_track__video19.mp4",
    "data/demo/halmstad__V_DRONE_035.mp4",
    "data/demo/halmstad__V_BIRD_020.mp4",
]
OUT = ROOT / "outputs" / "video" / "tracks"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=R2, help="poids Ultralytics")
    p.add_argument("--imgsz", type=int, default=1280, help="taille d'inférence (1280 pour R2, 640 pour R1)")
    p.add_argument("--sources", nargs="+", default=DEMO_SOURCES, help="vidéos à traiter")
    p.add_argument("--device", default=None, help="0, cpu… (défaut : GPU si disponible)")
    p.add_argument("--conf", type=float, default=None, help="seuil du détecteur (défaut : celui de demo.py)")
    p.add_argument("--lock-conf", type=float, default=None, help="confiance d'acquisition (défaut : demo.py)")
    p.add_argument("--tag", default="r2", help="suffixe du fichier de sortie (<source>__<tag>.json)")
    p.add_argument("--max-frames", type=int, default=0, help="limite de frames par source (0 = tout ; tests)")
    p.add_argument("--force", action="store_true", help="recalculer même si le JSON existe")
    return p.parse_args()


def r1(v: float) -> float:
    return round(float(v), 1)


def dump(model, source: str, args, out_path: Path) -> None:
    # Réglages par défaut de la démo (verrou, tracker, conf) : source de vérité = demo.py.
    argv = ["--source", source, "--weights", args.weights, "--imgsz", str(args.imgsz)]
    if args.conf is not None:
        argv += ["--conf", str(args.conf)]
    if args.lock_conf is not None:
        argv += ["--lock-conf", str(args.lock_conf)]
    d = demo.parse_args(argv)
    cap, is_file, src_fps, n_total, frame_idx = demo.open_source(d)
    lock = LockOn(lock_conf=d.lock_conf, coast_frames=d.coast_frames, min_frames=d.lock_min_frames,
                  reacquire_radius=d.reacquire_radius)  # fmt: skip
    kw = dict(persist=True, tracker=d.tracker, imgsz=d.imgsz, conf=d.conf, classes=None,
              device=args.device, verbose=False)  # fmt: skip
    ema = demo.Ema(alpha=0.1)
    frames, proc, model_ms = [], [], []
    hardware, label, size = "?", "?", None
    n = 0
    t_log = time.perf_counter()
    while not args.max_frames or n < args.max_frames:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        result = model.track(frame, **kw)[0]
        tracks, dets = demo.extract_tracks(result)
        out = lock.update(tracks, w, h)
        dt = time.perf_counter() - t0
        if n == 0:
            hardware, label = demo.describe_device(model)
            size = (w, h)
        else:
            ema.update(dt)
            proc.append(dt)
            model_ms.append(sum(result.speed.values()))
        t = out.target
        frames.append({
            "i": frame_idx,
            "fps": round(ema.fps, 2) if ema.fps else None,
            "state": out.state.value,
            "lost": out.lost_frames,
            "new": out.new_lock,
            "n": out.n_tracks,
            "tgt": None if t is None else [t.track_id, r1(t.x1), r1(t.y1), r1(t.x2), r1(t.y2), round(t.conf, 3)],
            "aim": None if out.aim is None else [round(out.aim.dx, 4), round(out.aim.dy, 4),
                                                 round(out.aim.box_area_ratio, 7)],
            "tracks": [[k.track_id, r1(k.x1), r1(k.y1), r1(k.x2), r1(k.y2), round(k.conf, 3), k.cls] for k in tracks],
            "dets": [[r1(a), r1(b), r1(c), r1(e), round(float(cf), 3), int(cl)] for a, b, c, e, cf, cl in dets],
        })  # fmt: skip
        n += 1
        frame_idx += 1
        if time.perf_counter() - t_log > 5:
            t_log = time.perf_counter()
            print(f"[dump] {Path(source).name} frame {frame_idx}/{n_total} FPS {ema.fps or 0:.1f} {out.state.value}")
    cap.release()
    counts = {s.value: sum(f["state"] == s.value for f in frames) for s in LockState}
    meta = {
        "source": source, "weights": args.weights, "imgsz": args.imgsz, "tracker": d.tracker, "conf": d.conf,
        "lock_conf": d.lock_conf, "lock_min_frames": d.lock_min_frames, "coast_frames": d.coast_frames,
        "reacquire_radius": d.reacquire_radius, "hardware": hardware, "device_label": label,
        "width": size[0] if size else None, "height": size[1] if size else None, "src_fps": src_fps,
        "n_frames": n, "pipeline_fps_mean": round(len(proc) / sum(proc), 2) if proc else None,
        "model_ms_mean": round(sum(model_ms) / len(model_ms), 2) if model_ms else None,
        "state_counts": counts, "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }  # fmt: skip
    out_path.write_text(json.dumps({"meta": meta, "frames": frames}, separators=(",", ":")))
    print(f"[dump] {out_path} : {n} frames, {meta['pipeline_fps_mean']} FPS pipeline ({hardware}), états {counts}")


def main() -> None:
    args = parse_args()
    from bench import tune_runtime  # comme demo.py : ONNX Runtime sans attente active
    from ultralytics import YOLO

    tune_runtime()

    OUT.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        out_path = OUT / f"{Path(source).stem}__{args.tag}.json"
        if out_path.exists() and not args.force:
            print(f"[dump] {out_path} existe déjà (--force pour recalculer)")
            continue
        model = YOLO(args.weights)  # un modèle par source : tracker et chauffe repartent de zéro
        dump(model, source, args, out_path)


if __name__ == "__main__":
    main()
