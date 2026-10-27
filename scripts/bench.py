"""Benchmark de latence / FPS de modèles YOLO (PyTorch .pt, ONNX .onnx, TensorRT .engine).

Pour chaque couple (poids, imgsz), batch 1, sur une image fixe (``--image``, par défaut la
1re image du split test drones, sinon bus.jpg d'Ultralytics) gardée en mémoire :
``--warmup`` prédictions de chauffe puis ``--iters`` prédictions mesurées.

Mesures (médiane et p90, en ms) :
    inférence   passe avant du modèle seule (``result.speed["inference"]`` d'Ultralytics,
                synchronisée CUDA) → **FPS modèle** = 1000 / médiane ;
    predict     appel ``model.predict`` complet sur l'image décodée : letterbox +
                normalisation + inférence + NMS + boîtes ramenées sur CPU → FPS predict.
Avec ``--video`` : FPS **de bout en bout** de la boucle de la démo (``scripts/demo.py``) :
lecture de la frame + ``model.track`` (détection + ByteTrack) + verrouillage
(``scripts/lockon.py``), sans affichage ni écriture ; ``--video-frames`` frames après
``--video-warmup`` frames de chauffe ; FPS = frames / temps total.

Équité : par défaut l'entrée est une letterbox **carrée** imgsz×imgsz pour tous les formats
(comme un export à taille fixe). ``--rect`` autorise la letterbox rectangulaire minimale
d'Ultralytics (.pt et ONNX dynamique seulement : moins de pixels, donc plus rapide). La
forme d'entrée réelle est notée dans le tableau. Un export à taille fixe ne tourne qu'à
son imgsz : les autres combinaisons sont sautées. ONNX Runtime tourne sans attente active
de ses threads (sinon il se bat avec les threads PyTorch d'Ultralytics et les deux
ralentissent, voir ``tune_runtime``) ; ``--ort-spinning`` garde le réglage d'origine.

Le matériel et les versions (CPU, GPU, pilote, alimentation secteur/batterie, torch,
CUDA, ONNX Runtime, TensorRT…) sont écrits dans la sortie, ainsi que l'occupation du GPU
par d'autres processus au lancement. Pendant chaque mesure GPU, ``nvidia-smi pmon`` surveille
l'activité des autres processus : une ligne mesurée pendant qu'un autre processus calculait est
marquée † (chiffre pessimiste) ; ``--wait-idle N`` attend N s de calme avant chaque mesure.

Sorties : ``runs/bench/<name>.json`` (tout) et ``runs/bench/<name>.md`` (tableau Markdown).

Usage (depuis la racine du dépôt) :
    .venv/bin/python scripts/bench.py --weights yolo11n.pt runs/export/yolo11n/yolo11n_640.onnx \\
        --imgsz 640 --device cpu --name cpu_yolo11n_640
    .venv/bin/python scripts/bench.py --weights best.pt best_1280.onnx best_1280_fp16.engine \\
        --imgsz 1280 --device 0 --video data/samples/blm_drone_training.webm --name gpu_best_1280
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))  # avant l'import d'ultralytics
sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True, help="poids à comparer (.pt, .onnx, .engine)")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640], help="taille(s) d'entrée (défaut : 640)")
    ap.add_argument("--device", default="cpu", help="cpu, 0, cuda:0… (défaut : cpu)")
    ap.add_argument(
        "--pt-precision",
        nargs="+",
        choices=["fp32", "fp16"],
        default=["fp32"],
        help="précision(s) PyTorch pour les .pt (fp16 : GPU uniquement) (défaut : fp32)",
    )
    ap.add_argument("--half", action="store_true", help="raccourci pour --pt-precision fp16")
    ap.add_argument("--image", default=None, help="image de mesure (défaut : 1re image du split test)")
    ap.add_argument("--conf", type=float, default=0.25, help="seuil de confiance (influe sur la NMS)")
    ap.add_argument("--rect", action="store_true", help="letterbox rectangulaire minimale (.pt / ONNX dynamique)")
    ap.add_argument("--warmup", type=int, default=20, help="prédictions de chauffe (défaut : 20)")
    ap.add_argument("--iters", type=int, default=200, help="prédictions mesurées (défaut : 200)")
    ap.add_argument("--video", default=None, help="vidéo pour le FPS de bout en bout (lecture + inférence + suivi)")
    ap.add_argument("--video-frames", type=int, default=300, help="frames mesurées sur --video (défaut : 300)")
    ap.add_argument("--video-warmup", type=int, default=10, help="frames de chauffe sur --video (défaut : 10)")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="tracker Ultralytics pour --video")
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="threads CPU max (torch, OpenCV, ONNX Runtime) (défaut : réglage des bibliothèques)",
    )
    ap.add_argument(
        "--ort-spinning",
        action="store_true",
        help="garder l'attente active des threads ONNX Runtime (réglage d'origine d'Ultralytics)",
    )
    ap.add_argument(
        "--wait-idle",
        type=int,
        default=0,
        help="avant chaque mesure GPU, attendre N s sans calcul GPU d'un autre processus (0 = non)",
    )
    ap.add_argument("--name", default=None, help="nom de la sortie runs/bench/<name> (défaut : bench_<date>)")
    return ap.parse_args(argv)


# --- Utilitaires -----------------------------------------------------------------------


def resolve(p: str | Path) -> Path:
    """Chemin relatif → relatif à la racine du dépôt (sauf s'il existe tel quel)."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (ROOT / p).resolve()


def rel(p: str | Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def stats(ms: list[float]) -> dict:
    a = np.asarray(ms, dtype=float)
    return {
        "median_ms": round(float(np.median(a)), 3),
        "p90_ms": round(float(np.percentile(a, 90)), 3),
        "mean_ms": round(float(a.mean()), 3),
        "min_ms": round(float(a.min()), 3),
        "n": int(a.size),
    }


def default_image() -> Path:
    test_dir = ROOT / "data" / "yolo" / "test" / "images"
    if test_dir.is_dir():
        imgs = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if imgs:
            return imgs[0]
    from ultralytics.utils import ASSETS

    return Path(ASSETS) / "bus.jpg"


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def power_source() -> str | None:
    """« secteur » / « batterie » (portables Linux), None si inconnu."""
    for p in Path("/sys/class/power_supply").glob("*"):
        try:
            if (p / "type").read_text().strip() == "Mains":
                return "secteur" if (p / "online").read_text().strip() == "1" else "batterie"
        except OSError:
            continue
    return None


def hardware_info(device: str) -> dict:
    """Matériel et versions logicielles, pour savoir sur quoi les chiffres ont été mesurés."""
    import importlib

    import cv2
    import torch

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "cpu": cpu,
        "cpu_logical_cores": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "power": power_source(),
        "loadavg_1m_at_start": round(os.getloadavg()[0], 2),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "opencv": cv2.__version__,
        "device": device,
    }
    for mod in ("ultralytics", "onnxruntime", "tensorrt"):
        try:
            info[mod] = importlib.import_module(mod).__version__
        except Exception:
            info[mod] = None
    if info["onnxruntime"]:
        import onnxruntime as ort

        info["onnxruntime_providers"] = ort.get_available_providers()
    if device != "cpu" and torch.cuda.is_available():
        idx = torch.device(f"cuda:{device}" if device.isdigit() else device).index or 0
        q = _run(["nvidia-smi", f"--id={idx}", "--format=csv,noheader,nounits",
                  "--query-gpu=name,driver_version,memory.total,power.limit,utilization.gpu,memory.used"])  # fmt: skip
        info["gpu"] = torch.cuda.get_device_name(idx)
        if q:
            name, driver, mem, plim, util, used = (s.strip() for s in q.split(","))
            info.update(gpu_driver=driver, gpu_mem_mib=mem, gpu_power_limit_w=plim,
                        gpu_util_at_start_pct=util, gpu_mem_used_at_start_mib=used)  # fmt: skip
        procs = _run(["nvidia-smi", f"--id={idx}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
        others = [ln for ln in (procs or "").splitlines() if ln.strip() and int(ln.split(",")[0]) != os.getpid()]
        info["gpu_other_processes"] = others
        if others:
            print(f"[bench] autres processus présents sur le GPU {others} : leur activité est surveillée (pmon)")
    return info


def hw_label(info: dict) -> str:
    if info.get("gpu"):
        return info["gpu"].replace("NVIDIA ", "").replace("GeForce ", "")
    return info["cpu"].replace(" with Radeon Graphics", "")


def model_meta(path: Path) -> dict:
    """Métadonnées Ultralytics d'un export (imgsz, dynamic, args d'export), {} pour un .pt."""
    if path.suffix == ".pt":
        return {}
    from ultralytics.nn.backends.base import BaseBackend

    meta = BaseBackend.read_metadata(path) or {}
    import ast

    for k in ("imgsz", "args"):
        if isinstance(meta.get(k), str):
            meta[k] = ast.literal_eval(meta[k])
    return meta


def export_precision(path: Path, meta: dict, half: bool) -> str:
    if path.suffix == ".pt":
        return "FP16" if half else "FP32"
    a = meta.get("args", {})
    q = a.get("quantize")
    if q in (8, "8", "int8") or a.get("int8"):
        return "INT8"
    if q in (16, "16", "fp16") or a.get("half"):
        return "FP16"
    return "FP32"


def runtime_label(model) -> str:
    """Runtime effectivement utilisé (lu sur l'AutoBackend après la 1re prédiction)."""
    be = model.predictor.model
    fmt = be.format
    if fmt == "pt":
        return f"PyTorch {'CUDA' if be.device.type == 'cuda' else 'CPU'}"
    if fmt == "onnx":
        sess = getattr(be.backend, "session", None)
        prov = sess.get_providers()[0] if sess else "?"
        return f"ONNX Runtime {prov.replace('ExecutionProvider', '')}"
    if fmt == "engine":
        return "TensorRT"
    return fmt


def tune_runtime(threads: int | None = None, ort_spinning: bool = False) -> None:
    """Règle les threads CPU (PyTorch, OpenCV, ONNX Runtime) de ce processus.

    ``threads`` : nombre maximal de threads de calcul (None = défaut des bibliothèques).
    ``ort_spinning=False`` : sessions ONNX Runtime sans attente active des threads intra-op.
    Par défaut, les threads d'ORT tournent à vide (« spinning ») après chaque inférence et
    volent le CPU aux threads PyTorch du pré/post-traitement d'Ultralytics dans le même
    processus : sur CPU, l'inférence ONNX et la letterbox ralentissent toutes deux (~×2
    observé sur Ryzen 7 5800H). Ultralytics ne permet pas de passer des ``SessionOptions`` :
    on les fournit en enveloppant le constructeur de son backend ONNX (ce processus seulement).
    """
    import cv2
    import onnxruntime as ort
    import torch
    from ultralytics.nn.backends.onnx import ONNXBackend

    if threads:
        import ultralytics.utils.torch_utils as tu

        tu.NUM_THREADS = threads  # select_device() d'Ultralytics refait torch.set_num_threads(NUM_THREADS)
        torch.set_num_threads(threads)
        cv2.setNumThreads(threads)
    if not threads and ort_spinning:
        return
    init = ONNXBackend.__init__

    def patched(self, weight, device, fp16=False, format="onnx", session_options=None):
        if session_options is None:
            session_options = ort.SessionOptions()
            if threads:
                session_options.intra_op_num_threads = threads
            if not ort_spinning:
                session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        init(self, weight, device, fp16, format, session_options)

    ONNXBackend.__init__ = patched


def input_shape(img: np.ndarray, imgsz: int, auto: bool) -> tuple[int, int]:
    from ultralytics.data.augment import LetterBox

    return LetterBox(imgsz, auto=auto, stride=32)(image=img).shape[:2]


# --- Mesures ---------------------------------------------------------------------------


def bench_image(model, img: np.ndarray, imgsz: int, args, kw: dict) -> dict:
    for _ in range(args.warmup):
        model.predict(img, **kw)
    infer, pre, post, wall = [], [], [], []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        r = model.predict(img, **kw)[0]
        _ = r.boxes.xyxy.cpu()  # boîtes disponibles côté CPU (synchronise le GPU)
        wall.append((time.perf_counter() - t0) * 1000)
        infer.append(r.speed["inference"])
        pre.append(r.speed["preprocess"])
        post.append(r.speed["postprocess"])
    return {"inference": stats(infer), "preprocess": stats(pre), "postprocess": stats(post), "predict": stats(wall)}


def bench_video(weights: Path, imgsz: int, args, kw: dict) -> dict:
    """Boucle de la démo sans affichage : lecture + model.track + extraction des pistes + verrouillage."""
    import cv2
    from demo import extract_tracks
    from demo import parse_args as demo_args
    from lockon import LockOn
    from ultralytics import YOLO

    d = demo_args(["--source", args.video])  # mêmes réglages de verrouillage que la démo
    lock = LockOn(lock_conf=d.lock_conf, coast_frames=d.coast_frames, min_frames=d.lock_min_frames,
                  reacquire_radius=d.reacquire_radius)  # fmt: skip
    model = YOLO(str(weights), task="detect")
    cap = cv2.VideoCapture(str(resolve(args.video)))
    if not cap.isOpened():
        raise SystemExit(f"impossible d'ouvrir la vidéo : {args.video}")
    track_kw = {**kw, "persist": True, "tracker": args.tracker}
    times, size = [], None
    try:
        for i in range(args.video_warmup + args.video_frames):
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            result = model.track(frame, **track_kw)[0]
            tracks, _ = extract_tracks(result)
            lock.update(tracks, w, h)
            dt = time.perf_counter() - t0
            if i >= args.video_warmup:
                times.append(dt * 1000)
                size = (w, h)
    finally:
        cap.release()
    if not times:
        raise SystemExit(f"vidéo trop courte pour --video-warmup {args.video_warmup} : {args.video}")
    s = stats(times)
    s["fps"] = round(len(times) / (sum(times) / 1000), 1)
    s["frame_size"] = f"{size[0]}x{size[1]}"
    return s


def model_label(path: Path) -> str:
    """runs/train/<run>/weights/best.pt → <run>/best.pt ; sinon le nom du fichier."""
    return f"{path.parent.parent.name}/{path.name}" if path.parent.name == "weights" else path.name


class GpuWatch:
    """Activité GPU des *autres* processus pendant une mesure (``nvidia-smi pmon``, 1 échantillon/s).

    Le GPU de la machine est partagé (autres projets, serveurs d'inférence…) : une mesure
    pendant laquelle un autre processus calcule est faussée. ``result`` donne le nombre
    d'échantillons où un autre processus avait une utilisation SM > 0 et le maximum observé.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled and bool(_run(["nvidia-smi", "-L"]))
        self.proc, self.result = None, None

    def __enter__(self):
        if self.enabled:
            self.proc = subprocess.Popen(["nvidia-smi", "pmon", "-s", "u", "-d", "1"], stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True)  # fmt: skip
        return self

    def __exit__(self, *exc):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            out = self.proc.communicate(timeout=5)[0]
        except subprocess.TimeoutExpired:
            self.proc.kill()
            out = self.proc.communicate()[0]
        busy = {}
        for line in out.splitlines():
            f = line.split()
            if len(f) < 4 or line.startswith("#") or not f[1].isdigit() or int(f[1]) == os.getpid():
                continue
            if f[3].isdigit() and int(f[3]) > 0:
                busy.setdefault(f[-1], []).append(int(f[3]))
        self.result = {
            "other_busy_samples": sum(len(v) for v in busy.values()),
            "other_sm_max_pct": max((max(v) for v in busy.values()), default=0),
            "other_processes": sorted(busy),
        }


def wait_gpu_idle(quiet_s: int, timeout_s: int = 900) -> float:
    """Attend ``quiet_s`` secondes consécutives sans calcul GPU d'un autre processus ; renvoie l'attente (s)."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        with GpuWatch(True) as w:
            time.sleep(quiet_s)
        if not w.enabled or w.result["other_busy_samples"] == 0:
            return time.time() - t0
        busy = f"{w.result['other_processes']} (SM max {w.result['other_sm_max_pct']} %)"
        print(f"[bench] GPU occupé par {busy} : attente…", flush=True)
    print(f"[bench] ATTENTION : GPU toujours occupé après {timeout_s} s, mesure quand même")
    return time.time() - t0


def run_one(weights: Path, imgsz: int, img: np.ndarray, args, info: dict, half: bool = False) -> dict | None:
    from ultralytics import YOLO

    meta = model_meta(weights)
    dynamic = bool(meta.get("args", {}).get("dynamic", False))
    if meta.get("imgsz") and not dynamic and max(meta["imgsz"]) != imgsz:
        print(f"[bench] {rel(weights)} : export fixe {meta['imgsz']} → imgsz {imgsz} sauté")
        return None
    if weights.suffix == ".engine" and args.device == "cpu":
        print(f"[bench] {rel(weights)} : TensorRT exige un GPU → sauté sur CPU")
        return None

    auto = args.rect and (weights.suffix == ".pt" or dynamic)
    kw = dict(imgsz=imgsz, conf=args.conf, device=args.device, rect=args.rect, verbose=False)
    if half and weights.suffix == ".pt":
        kw["quantize"] = 16
    gpu = args.device != "cpu"
    if args.wait_idle:  # un calcul GPU d'un autre processus charge aussi le CPU
        wait_gpu_idle(args.wait_idle)
    model = YOLO(str(weights), task="detect")
    print(f"[bench] {rel(weights)} imgsz {imgsz} : {args.warmup} chauffe + {args.iters} mesures…", flush=True)
    with GpuWatch(gpu) as watch:
        res = bench_image(model, img, imgsz, args, kw)
    h, w = input_shape(img, imgsz, auto)
    row = {
        "weights": rel(weights),
        "label": model_label(weights),
        "format": weights.suffix.lstrip("."),
        "runtime": runtime_label(model),
        "precision": export_precision(weights, meta, half),
        "imgsz": imgsz,
        "input": f"{w}x{h}",
        "dynamic": dynamic,
        "hardware": "CPU " + info["cpu"] if args.device == "cpu" else "GPU " + info.get("gpu", "?"),
        **res,
        "fps_model": round(1000 / res["inference"]["median_ms"], 1),
        "fps_predict": round(1000 / res["predict"]["median_ms"], 1),
    }
    del model
    row["gpu_others"] = watch.result
    if args.video:
        print(f"[bench] {rel(weights)} imgsz {imgsz} : vidéo {args.video} ({args.video_frames} frames)…", flush=True)
        with GpuWatch(gpu) as watch:
            row["video"] = bench_video(weights, imgsz, args, kw)
        row["video"]["gpu_others"] = watch.result
    row["loadavg_1m_end"] = round(os.getloadavg()[0], 2)
    busy = [r for r in (row["gpu_others"], row.get("video", {}).get("gpu_others")) if r and r["other_busy_samples"]]
    if busy:
        print(f"[bench]   ATTENTION : autre processus actif sur le GPU pendant la mesure {busy} → ligne marquée †")
    print(f"[bench]   {row['runtime']} {row['precision']} entrée {row['input']} : inférence "
          f"{res['inference']['median_ms']:.2f} ms (p90 {res['inference']['p90_ms']:.2f}) = {row['fps_model']} FPS ; "
          f"predict {res['predict']['median_ms']:.2f} ms = {row['fps_predict']} FPS"
          + (f" ; vidéo {row['video']['fps']} FPS" if args.video else ""))  # fmt: skip
    return row


# --- Sorties ---------------------------------------------------------------------------


def markdown(rows: list[dict], info: dict, args, image: Path) -> str:
    hw = f"**{info['gpu']}**" if info.get("gpu") else f"**{info['cpu']}** ({info['torch_threads']} threads torch)"
    lines = [
        f"Matériel : {hw} — {info['cpu']} ({info['cpu_logical_cores']} threads logiques) — alimentation "
        f"{info.get('power') or '?'} — charge système au lancement {info['loadavg_1m_at_start']} — {info['os']}",
        f"Versions : torch {info['torch']} (CUDA {info['cuda']}), ultralytics {info['ultralytics']}, "
        f"onnxruntime {info['onnxruntime']}, TensorRT {info['tensorrt'] or '—'} — {info['date']}",
        f"Image de mesure : `{rel(image)}`, batch 1, conf {args.conf}, {args.warmup} chauffe + {args.iters} "
        f"mesures, letterbox {'rectangulaire' if args.rect else 'carrée'}, "
        f"spinning ORT {'activé' if args.ort_spinning else 'désactivé'}"
        + (f", threads CPU limités à {args.threads}" if args.threads else "")
        + (f" ; vidéo `{args.video}` ({args.video_frames} frames, tracker {args.tracker})" if args.video else ""),
    ]
    if info.get("gpu"):
        others = len(info.get("gpu_other_processes") or [])
        shared = f" — {others} autre(s) processus présent(s) sur le GPU, activité surveillée (†)" if others else ""
        lines.append(f"GPU au lancement : {info.get('gpu_util_at_start_pct', '?')} % d'utilisation, "
                     f"{info.get('gpu_mem_used_at_start_mib', '?')} Mio occupés, pilote {info.get('gpu_driver', '?')}, "
                     f"limite {info.get('gpu_power_limit_w', '?')} W{shared}")  # fmt: skip
    lines += ["", "| Modèle | Runtime | Précision | Entrée | Inférence méd. / p90 (ms) | FPS modèle "
              "| Predict méd. / p90 (ms) | FPS predict |" + (" FPS vidéo bout en bout |" if args.video else ""),
              "|---|---|---|---|---|---|---|---|" + ("---|" if args.video else "")]  # fmt: skip
    contaminated = False
    for r in rows:
        dag = lambda g: " †" if g and g["other_busy_samples"] else ""  # noqa: E731
        mark_img, mark_vid = dag(r.get("gpu_others")), dag(r.get("video", {}).get("gpu_others"))
        contaminated |= bool(mark_img or mark_vid)
        line = (f"| `{r['label']}`{mark_img} | {r['runtime']} | {r['precision']} | {r['input']} "
                f"| {r['inference']['median_ms']:.2f} / {r['inference']['p90_ms']:.2f} | {r['fps_model']} "
                f"| {r['predict']['median_ms']:.2f} / {r['predict']['p90_ms']:.2f} | {r['fps_predict']} |")  # fmt: skip
        if args.video:
            v = r.get("video")
            line += f" {v['fps']} ({v['frame_size']}){mark_vid} |" if v else " — |"
        lines.append(line)
    if contaminated:
        lines += ["", "† un autre processus calculait sur le GPU pendant cette mesure (nvidia-smi pmon) : "
                  "chiffre pessimiste, à refaire."]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import cv2

    weights = []
    for w in args.weights:
        p = resolve(w)
        if not p.exists():
            raise SystemExit(f"poids introuvables : {w}")
        weights.append(p)
    image = resolve(args.image) if args.image else default_image()
    img = cv2.imread(str(image))
    if img is None:
        raise SystemExit(f"image illisible : {image}")
    args.device = str(args.device)
    if args.half:
        args.pt_precision = ["fp16"]
    if "fp16" in args.pt_precision and args.device == "cpu":
        print("[bench] PyTorch FP16 ignoré sur CPU (Ultralytics calcule en FP32 sur CPU)")
        args.pt_precision = ["fp32"]
    name = args.name or f"bench_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = ROOT / "runs" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    tune_runtime(args.threads, args.ort_spinning)
    info = hardware_info(args.device)
    info["threads_limit"] = args.threads
    print(f"[bench] matériel : {hw_label(info)} (device {args.device}) ; image {rel(image)} "
          f"{img.shape[1]}x{img.shape[0]}")  # fmt: skip
    rows = []
    for imgsz in args.imgsz:
        for w in weights:
            for prec in args.pt_precision if w.suffix == ".pt" else ["fp32"]:
                row = run_one(w, imgsz, img, args, info, half=prec == "fp16")
                if row:
                    rows.append(row)
    if not rows:
        raise SystemExit("aucune combinaison poids / imgsz mesurable")

    md = markdown(rows, info, args, image)
    (out_dir / f"{name}.json").write_text(json.dumps(
        {"name": name, "hardware": info, "settings": {**vars(args), "image": rel(image)}, "results": rows},
        indent=2, ensure_ascii=False))  # fmt: skip
    (out_dir / f"{name}.md").write_text(md)
    print("\n" + md)
    print(f"[bench] sorties : runs/bench/{name}.json, runs/bench/{name}.md")


if __name__ == "__main__":
    main()
