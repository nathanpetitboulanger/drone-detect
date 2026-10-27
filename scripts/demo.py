"""Démo vidéo « verrouillage » : détection + suivi multi-objets + erreur de visée.

Lit une vidéo (fichier, URL ou webcam) frame par frame, applique un modèle Ultralytics
avec suivi (``model.track``, ByteTrack ou BoT-SORT), verrouille une cible (voir
``scripts/lockon.py``) et affiche un overlay de type « lock-on » pour le guidage terminal
d'un intercepteur. **Aucun code de contrôle de vol** : la démo s'arrête
au vecteur d'erreur de visée (dx, dy), exporté en CSV.

À lancer depuis la racine du dépôt, par exemple (modèle COCO, le drone sort souvent en
``airplane``, ``bird`` ou ``kite``) :

    .venv/bin/python scripts/demo.py --source data/samples/blm_drone_training.webm \\
        --device cpu --classes airplane bird kite --lock-conf 0.25 \\
        --save outputs/demo/blm.mp4 --csv outputs/demo/blm.csv --gif outputs/demo/blm.gif

    .venv/bin/python scripts/demo.py --source 0 --show            # webcam
    .venv/bin/python scripts/demo.py --weights best.onnx --source video.mp4 --save out.mp4

Fonctionne avec n'importe quels poids Ultralytics (``.pt``, ``.onnx``, ``.engine``…).
Pour un ``.onnx`` / ``.engine`` exporté en taille fixe, passer le même ``--imgsz``.
Avec un ``.onnx``, l'attente active des threads ONNX Runtime est désactivée : sur CPU, elle
ralentit l'inférence et le pré-traitement PyTorch (voir ``tune_runtime`` dans ``bench.py``).

FPS mesurés
    « FPS pipeline » : 1 / (lecture de la frame + inférence + suivi + verrouillage), en
    moyenne glissante exponentielle (EMA) à l'écran et dans le CSV ; moyenne finale hors
    1re frame (chauffe du modèle). C'est la cadence qu'aurait la boucle de guidage.
    « Débit global » : frames / temps total, overlay et écritures (mp4, GIF, CSV) compris.

CSV (``--csv``, une ligne par frame) — l'entrée qu'un contrôleur de guidage consommerait :
    frame           index de la frame dans la source (0 = première frame de la vidéo)
    t_s             horodatage (s) : frame / fps source pour un fichier, temps écoulé
                    depuis le début pour une webcam ou un flux
    fps             FPS pipeline (EMA) à cette frame (vide pour la 1re frame)
    state           SEARCHING | LOCKED | COASTING
    track_id        ID (tracker) de la cible verrouillée
    conf            confiance de la détection de la cible
    cx, cy          centre de la boîte de la cible (px, origine en haut à gauche)
    w, h            largeur et hauteur de la boîte (px)
    dx, dy          erreur de visée normalisée dans [-1, 1] :
                    dx = (cx - W/2) / (W/2), dy = (cy - H/2) / (H/2) ;
                    dx > 0 : cible à droite du centre, dy > 0 : cible en bas
    box_area_ratio  (w * h) / (W * H), proxy de distance (augmente quand la cible approche)
    n_tracks        nombre de pistes confirmées dans la frame
    Les colonnes de la cible (track_id … box_area_ratio) sont vides en SEARCHING ; en
    COASTING elles reprennent la dernière observation de la cible (donnée périmée).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))  # avant l'import d'ultralytics
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lockon import LockOn, LockOutput, LockState, Track  # noqa: E402

# Couleurs de l'overlay (BGR).
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
STATE_COLORS = {
    LockState.SEARCHING: (0, 220, 255),  # jaune
    LockState.LOCKED: (40, 40, 255),  # rouge
    LockState.COASTING: (0, 150, 255),  # orange
}
TRACK_COLOR = (255, 200, 0)  # bleu clair : pistes non verrouillées
DET_COLOR = (170, 170, 170)  # gris : détections pas encore confirmées par le tracker
OVERLAY_COLORS = [BLACK, WHITE, TRACK_COLOR, DET_COLOR, *STATE_COLORS.values()]
FONT = cv2.FONT_HERSHEY_SIMPLEX

EPILOG = """\
exemples :
  %(prog)s --source data/samples/blm_drone_training.webm --device cpu \\
      --classes airplane bird kite --lock-conf 0.25 \\
      --save outputs/demo/blm.mp4 --csv outputs/demo/blm.csv --gif outputs/demo/blm.gif
  %(prog)s --source 0 --show                      # webcam, fenêtre (q / Échap pour quitter)
  %(prog)s --weights runs/detect/train/weights/best.onnx --source video.mp4 --save out.mp4

états du verrou : SEARCHING (pas de cible) -> LOCKED (cible suivie) -> COASTING (cible
perdue depuis <= --coast-frames frames, dernière position gardée) -> SEARCHING.
Colonnes du CSV et convention dx/dy : voir la docstring en tête de scripts/demo.py.
"""


# --- Arguments -------------------------------------------------------------------------


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Démo vidéo « verrouillage » : détection + suivi + erreur de visée (dx, dy) "
        "pour le guidage terminal par vision. Aucun code de contrôle de vol.",
        epilog=EPILOG,
        formatter_class=_HelpFormatter,
    )
    g = p.add_argument_group("modèle et suivi")
    g.add_argument("--weights", default="yolo11n.pt", help="poids Ultralytics (.pt, .onnx, .engine…)")
    g.add_argument("--source", required=True, help="fichier vidéo, URL, ou index de webcam (0, 1…)")
    g.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="config du tracker Ultralytics (bytetrack.yaml, botsort.yaml…) ou chemin vers un .yaml",
    )
    g.add_argument("--imgsz", type=int, default=640, help="taille d'inférence (960-1280 aide pour les petits drones)")
    g.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="seuil de confiance du détecteur (bas : le tracker exploite aussi les détections faibles)",
    )
    g.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="CLS",
        help="ne garder que ces classes (indices ou noms, ex. 'airplane bird kite' avec le modèle COCO)",
    )
    g.add_argument("--device", default=None, help="cpu, 0, cuda:0… (défaut : GPU si disponible)")
    g.add_argument("--half", action="store_true", help="inférence FP16 (GPU, poids .pt)")

    g = p.add_argument_group("verrouillage")
    g.add_argument("--lock-conf", type=float, default=0.5, help="confiance minimale pour acquérir une cible")
    g.add_argument(
        "--lock-min-frames",
        type=int,
        default=3,
        help="persistance minimale (frames consécutives >= --lock-conf) avant verrouillage",
    )
    g.add_argument(
        "--coast-frames", type=int, default=15, help="frames gardées en COASTING avant de relâcher la cible"
    )
    g.add_argument(
        "--reacquire-radius",
        type=float,
        default=0.0,
        help="en COASTING, re-verrouiller une piste à moins de ce rayon (unités dx/dy) de la dernière "
        "position (0 = désactivé)",
    )

    g = p.add_argument_group("entrées / sorties")
    g.add_argument("--save", type=Path, default=None, metavar="OUT.mp4", help="vidéo annotée (H.264 via ffmpeg)")
    g.add_argument("--csv", type=Path, default=None, metavar="OUT.csv", help="CSV par frame (erreur de visée)")
    g.add_argument("--gif", type=Path, default=None, metavar="OUT.gif", help="GIF annoté sous-échantillonné")
    g.add_argument("--gif-fps", type=float, default=8.0, help="images par seconde du GIF")
    g.add_argument("--gif-width", type=int, default=640, help="largeur du GIF (px)")
    g.add_argument("--gif-max-mb", type=float, default=8.0, help="taille visée du GIF (réduit si dépassée)")
    g.add_argument("--gif-max-frames", type=int, default=400, help="nombre maximal d'images du GIF")
    g.add_argument(
        "--show", action=argparse.BooleanOptionalAction, default=False, help="afficher une fenêtre (q / Échap)"
    )
    g.add_argument("--max-frames", type=int, default=0, help="nombre maximal de frames traitées (0 = tout)")
    g.add_argument("--start-frame", type=int, default=0, help="première frame à traiter")
    return p.parse_args(argv)


def parse_source(source: str) -> int | str:
    """Index de webcam si la source est un entier, sinon chemin ou URL."""
    return int(source) if source.isdigit() else source


def resolve_classes(tokens: list[str] | None, names: dict[int, str]) -> list[int] | None:
    """Convertit ``--classes`` (indices ou noms, séparés par espaces ou virgules) en indices."""
    if not tokens:
        return None
    lookup = {str(v).lower(): int(k) for k, v in names.items()}
    out = []
    for tok in tokens:
        for part in (s.strip() for s in tok.split(",")):
            if not part:
                continue
            if part.isdigit():
                out.append(int(part))
            elif part.lower() in lookup:
                out.append(lookup[part.lower()])
            else:
                raise SystemExit(f"classe inconnue : {part!r}. Classes du modèle : {dict(names)}")
    return sorted(set(out))


# --- Utilitaires -----------------------------------------------------------------------


class Ema:
    """Moyenne glissante exponentielle d'une durée ; ``fps`` = 1 / durée moyenne."""

    def __init__(self, alpha: float = 0.1):
        self.alpha, self.value = alpha, None

    def update(self, dt: float) -> float | None:
        self.value = dt if self.value is None else (1 - self.alpha) * self.value + self.alpha * dt
        return self.fps

    @property
    def fps(self) -> float | None:
        return 1.0 / self.value if self.value else None


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def describe_device(model) -> tuple[str, str]:
    """(description complète, étiquette courte pour l'overlay) du matériel d'inférence."""
    import torch

    dev = getattr(getattr(model, "predictor", None), "device", None)
    if dev is not None and getattr(dev, "type", "") == "cuda":
        name = torch.cuda.get_device_name(dev)
        return f"GPU {name} (CUDA, torch {torch.__version__})", name.replace("NVIDIA ", "").replace("GeForce ", "")
    return f"CPU {cpu_name()} ({torch.get_num_threads()} threads torch)", "CPU"


def extract_tracks(result) -> tuple[list[Track], list[tuple]]:
    """Pistes confirmées (avec ID) et détections sans ID d'un ``Results`` Ultralytics."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []
    xyxy = boxes.xyxy.cpu().numpy().astype(float)
    conf = boxes.conf.cpu().numpy().astype(float)
    cls = boxes.cls.cpu().numpy().astype(int)
    if boxes.id is None:  # tracker pas encore confirmé : détections brutes, non verrouillables
        return [], [(*xyxy[i], conf[i], cls[i]) for i in range(len(conf))]
    ids = boxes.id.cpu().numpy().astype(int)
    tracks = []
    for i in range(len(ids)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i])
        tracks.append(Track(int(ids[i]), x1, y1, x2, y2, float(conf[i]), int(cls[i])))
    return tracks, []


def _has_x264() -> bool:
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return "libx264" in out.stdout


class VideoSink:
    """Écrit un mp4 H.264 via ffmpeg (lisible par les navigateurs), sinon mp4v via OpenCV."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        w, h = size
        self.path, self.proc, self.writer = path, None, None
        if _has_x264():
            cmd = [
                "ffmpeg", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
            ]  # fmt: skip
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not self.writer.isOpened():
                raise SystemExit(f"impossible d'écrire la vidéo : {path}")

    def write(self, frame: np.ndarray) -> None:
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.writer.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            if self.proc.wait() != 0:
                print(f"[demo] attention : ffmpeg a échoué pour {self.path}", file=sys.stderr)
        elif self.writer is not None:
            self.writer.release()


def write_gif(frames: list[np.ndarray], path: Path, duration_ms: float) -> None:
    """GIF à palette globale (couleurs de l'overlay exactes), sans tramage pour rester léger."""
    from PIL import Image

    n_adaptive = 256 - len(OVERLAY_COLORS)
    step = max(1, len(frames) // 12)
    mosaic = Image.fromarray(np.concatenate(frames[::step][:12], axis=0))
    palette = (mosaic.quantize(colors=n_adaptive, method=Image.Quantize.MEDIANCUT).getpalette() or [])[: 3 * n_adaptive]
    palette += [0] * (3 * n_adaptive - len(palette))
    for b, g, r in OVERLAY_COLORS:
        palette += [r, g, b]
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette(palette)
    images = [Image.fromarray(f).quantize(palette=pal_img, dither=Image.Dither.NONE) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(round(duration_ms)), loop=0)


class GifRecorder:
    """Collecte des frames annotées sous-échantillonnées et réduites, puis écrit le GIF."""

    def __init__(self, path: Path, src_fps: float, gif_fps: float, width: int, max_mb: float, max_frames: int):
        self.path, self.width, self.max_mb, self.max_frames = path, width, max_mb, max_frames
        self.step = max(1, round(src_fps / gif_fps)) if gif_fps > 0 else 1
        self.duration_ms = 1000.0 * self.step / src_fps
        self.frames: list[np.ndarray] = []
        self.truncated = False

    def add(self, n: int, frame: np.ndarray) -> None:
        if n % self.step:
            return
        if len(self.frames) >= self.max_frames:
            self.truncated = True
            return
        h, w = frame.shape[:2]
        width = min(self.width, w)
        small = cv2.resize(frame, (width, max(2, round(h * width / w))), interpolation=cv2.INTER_AREA)
        self.frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    def save(self) -> str:
        if not self.frames:
            return f"{self.path} : aucune image, GIF non écrit"
        frames, duration = self.frames, self.duration_ms
        for attempt in range(5):
            write_gif(frames, self.path, duration)
            mb = self.path.stat().st_size / 1e6
            if mb <= self.max_mb or attempt == 4 or len(frames) < 4:
                break
            # Trop gros : on réduit la taille, puis (sous 320 px) le nombre d'images.
            h, w = frames[0].shape[:2]
            factor = 0.92 * math.sqrt(self.max_mb / mb)
            if w * factor >= 320:
                size = (int(w * factor), max(2, round(h * factor)))
                frames = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
            else:
                frames, duration = frames[::2], duration * 2
        h, w = frames[0].shape[:2]
        note = f", tronqué à {self.max_frames} images" if self.truncated else ""
        return f"{self.path} ({mb:.1f} Mo, {len(frames)} images {w}x{h} à {1000 / duration:.1f} img/s{note})"


# --- Overlay ---------------------------------------------------------------------------


def put_text(img, text, org, scale, color, thick=1):
    """Texte avec contour noir (lisible sur ciel clair comme sombre).

    Le contour est le texte noir décalé de quelques pixels, à la même épaisseur : avec OpenCV 5,
    une épaisseur plus forte élargit aussi l'avance des glyphes, et un contour « épais »
    débordait du texte (caractère fantôme en fin de ligne)."""
    x, y = int(org[0]), int(org[1])
    r = max(1, round(0.6 * thick))
    for ox in (-r, 0, r):
        for oy in (-r, 0, r):
            if ox or oy:
                cv2.putText(img, text, (x + ox, y + oy), FONT, scale, BLACK, thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def text_size(text, scale, thick):
    (w, h), base = cv2.getTextSize(text, FONT, scale, thick)
    return w, h + base


def shade(img, x1, y1, x2, y2, keep=0.45):
    """Assombrit un rectangle (fond des panneaux de texte)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 > x1 and y2 > y1:
        img[y1:y2, x1:x2] = (img[y1:y2, x1:x2] * keep).astype(np.uint8)


def draw_brackets(img, x1, y1, x2, y2, color, thick):
    """Coins de « verrouillage » autour d'un rectangle."""
    length = int(max(4, 0.3 * min(x2 - x1, y2 - y1)))
    for x, y, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (x, y), (x + sx * length, y), color, thick, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + sy * length), color, thick, cv2.LINE_AA)


def draw_reticle(img, s):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    r, gap, arm = int(16 * s), int(6 * s), int(30 * s)
    t = max(1, round(1.5 * s))
    for color, thick in ((BLACK, t + 2), (WHITE, t)):
        cv2.circle(img, (cx, cy), r, color, thick, cv2.LINE_AA)
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(img, (cx + sx * gap, cy + sy * gap), (cx + sx * arm, cy + sy * arm), color, thick, cv2.LINE_AA)


def draw_overlay(img, out: LockOutput, tracks, dets, names, fps, device_label, frame_idx, t_s, coast_frames):
    """Dessine l'overlay « lock-on » sur ``img`` (en place)."""
    h, w = img.shape[:2]
    s = max(0.5, min(w, h) / 720)  # échelle : 1 en 720p
    thin, thick = max(1, round(s)), max(2, round(3 * s))
    small = 0.5 * s
    target_id = out.target.track_id if out.target is not None else None
    color = STATE_COLORS[out.state]

    # Détections non confirmées et pistes non verrouillées : trait fin.
    for x1, y1, x2, y2, conf, cls in dets:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), DET_COLOR, thin, cv2.LINE_AA)
    for t in tracks:
        if t.track_id == target_id:
            continue
        cv2.rectangle(img, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), TRACK_COLOR, thin, cv2.LINE_AA)
        label = f"#{t.track_id} {names.get(t.cls, t.cls)} {t.conf:.2f}"
        put_text(img, label, (t.x1, max(t.y1 - 4 * s, 12 * s)), small, TRACK_COLOR, thin)

    draw_reticle(img, s)

    # Cible verrouillée : cadre épais (taille minimale pour rester visible), coins, ligne.
    if out.target is not None:
        t = out.target
        cx, cy = int(round(t.cx)), int(round(t.cy))
        cv2.line(img, (w // 2, h // 2), (cx, cy), color, max(1, round(1.5 * s)), cv2.LINE_AA)
        hw, hh = max(t.w / 2, 10 * s), max(t.h / 2, 10 * s)
        bx1, by1, bx2, by2 = int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)
        pad = int(max(8 * s, 0.25 * max(hw, hh)))
        if out.state is LockState.LOCKED:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, thick, cv2.LINE_AA)
        draw_brackets(img, bx1 - pad, by1 - pad, bx2 + pad, by2 + pad, color, thick)
        cv2.circle(img, (cx, cy), max(2, int(3 * s)), color, -1, cv2.LINE_AA)
        tag = "LOCK" if out.state is LockState.LOCKED else "COAST"
        label = f"{tag} #{t.track_id} {t.conf:.2f}"
        lw, lh = text_size(label, 0.7 * s, thin + 1)
        ly = by1 - pad - 6 * s
        if ly - lh < 0:
            ly = by2 + pad + lh + 6 * s
        lx = min(max(2, bx1 - pad), w - lw - 2)
        put_text(img, label, (lx, ly), 0.7 * s, color, thin + 1)

    # Panneau d'état (haut gauche).
    if out.state is LockState.LOCKED:
        state_txt = f"LOCKED #{target_id}"
    elif out.state is LockState.COASTING:
        state_txt = f"COASTING {out.lost_frames}/{coast_frames}"
    else:
        state_txt = "SEARCHING"
    if out.aim is not None:
        aim_txt = f"dx={out.aim.dx:+.3f} dy={out.aim.dy:+.3f}"
        size_txt = f"size={100 * out.aim.box_area_ratio:.2f}%  tracks={out.n_tracks}"
    else:
        aim_txt, size_txt = "dx=  --    dy=  --", f"tracks={out.n_tracks}"
    lines = [(state_txt, 1.1 * s, max(2, round(2 * s)), color), (aim_txt, 0.9 * s, max(1, round(2 * s)), WHITE)]
    lines.append((size_txt, 0.6 * s, thin, WHITE))
    m = int(10 * s)
    sizes = [text_size(txt, sc, th) for txt, sc, th, _ in lines]
    panel_w = max(sw for sw, _ in sizes) + 2 * m
    panel_h = sum(sh for _, sh in sizes) + m * (len(lines) + 1)
    shade(img, m // 2, m // 2, m // 2 + panel_w, m // 2 + panel_h)
    y = m // 2
    for (txt, sc, th, col), (_, sh) in zip(lines, sizes):
        y += m + sh
        put_text(img, txt, (m // 2 + m, y - sh // 4), sc, col, th)

    # FPS et matériel (haut droite).
    fps_txt = f"FPS {fps:.1f}" if fps else "FPS --"
    fw, fh = text_size(fps_txt, 0.9 * s, max(1, round(2 * s)))
    dw, dh = text_size(device_label, small, thin)
    pw = max(fw, dw) + 2 * m
    shade(img, w - pw - m // 2, m // 2, w - m // 2, m // 2 + fh + dh + 3 * m)
    put_text(img, fps_txt, (w - m // 2 - m - fw, m // 2 + m + fh), 0.9 * s, WHITE, max(1, round(2 * s)))
    put_text(img, device_label, (w - m // 2 - m - dw, m // 2 + 2 * m + fh + dh), small, WHITE, thin)

    # Index de frame (bas gauche).
    put_text(img, f"frame {frame_idx}  t={t_s:.2f}s", (m, h - m), small, WHITE, thin)


# --- Boucle principale -----------------------------------------------------------------

CSV_HEADER = ["frame", "t_s", "fps", "state", "track_id", "conf", "cx", "cy", "w", "h", "dx", "dy",
              "box_area_ratio", "n_tracks"]  # fmt: skip


def csv_row(frame_idx: int, t_s: float, fps: float | None, out: LockOutput) -> list:
    row = [frame_idx, f"{t_s:.3f}", f"{fps:.1f}" if fps else "", out.state.value]
    if out.target is None:
        return row + [""] * 9 + [out.n_tracks]
    t, a = out.target, out.aim
    return row + [
        t.track_id, f"{t.conf:.3f}", f"{t.cx:.1f}", f"{t.cy:.1f}", f"{t.w:.1f}", f"{t.h:.1f}",
        f"{a.dx:.4f}", f"{a.dy:.4f}", f"{a.box_area_ratio:.6f}", out.n_tracks,
    ]  # fmt: skip


def open_source(args) -> tuple[cv2.VideoCapture, bool, float, int, int]:
    """Ouvre la source et se place à ``--start-frame``. Renvoie (cap, is_file, fps, n_total, start)."""
    src = parse_source(args.source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"impossible d'ouvrir la source : {args.source}")
    is_file = isinstance(src, str) and Path(src).is_file()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps < 1000:
        fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    start = max(0, args.start_frame)
    if start and is_file:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    elif start:
        for _ in range(start):
            cap.grab()
    return cap, is_file, fps, n_total, start


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from bench import tune_runtime
    from ultralytics import YOLO

    tune_runtime()  # sessions ONNX Runtime sans spinning (sinon ~×3 plus lent sur CPU), threads inchangés
    model = YOLO(args.weights)
    names = {int(k): str(v) for k, v in model.names.items()}
    classes = resolve_classes(args.classes, names)
    cap, is_file, src_fps, n_total, frame_idx = open_source(args)
    lock = LockOn(
        lock_conf=args.lock_conf,
        coast_frames=args.coast_frames,
        min_frames=args.lock_min_frames,
        reacquire_radius=args.reacquire_radius,
    )
    track_kwargs = dict(
        persist=True, tracker=args.tracker, imgsz=args.imgsz, conf=args.conf, classes=classes,
        device=args.device, verbose=False,
    )  # fmt: skip
    if args.half:
        track_kwargs["quantize"] = 16

    for p in (args.save, args.csv, args.gif):
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(args.csv, "w", newline="") if args.csv else None
    csv_writer = csv.writer(csv_file) if csv_file else None
    if csv_writer:
        csv_writer.writerow(CSV_HEADER)
    gif = (
        GifRecorder(args.gif, src_fps, args.gif_fps, args.gif_width, args.gif_max_mb, args.gif_max_frames)
        if args.gif
        else None
    )
    sink, show = None, args.show
    if show:
        cv2.namedWindow("drone-detect lock-on", cv2.WINDOW_NORMAL)

    class_txt = f", classes={[names.get(c, c) for c in classes]}" if classes else ""
    print(f"[demo] poids={args.weights} tracker={args.tracker} imgsz={args.imgsz} conf={args.conf}{class_txt}")
    print(f"[demo] source={args.source} ({'fichier' if is_file else 'flux'}, {src_fps:.2f} fps"
          f"{f', {n_total} frames' if n_total else ''}), départ frame {frame_idx}")  # fmt: skip

    ema = Ema(alpha=0.1)
    proc_times, model_ms = [], []
    counts = {st: 0 for st in LockState}
    acquisitions, locked_ids = 0, []
    hardware, device_label, size = "?", "?", None
    n, interrupted = 0, False
    t_start = t_log = time.perf_counter()
    try:
        while not args.max_frames or n < args.max_frames:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            result = model.track(frame, **track_kwargs)[0]
            tracks, dets = extract_tracks(result)
            out = lock.update(tracks, w, h)
            dt = time.perf_counter() - t0

            if n == 0:  # 1re frame : chauffe du modèle, exclue des moyennes
                hardware, device_label = describe_device(model)
                size = (w, h)
                print(f"[demo] matériel : {hardware} ; image {w}x{h}")
            else:
                ema.update(dt)
                proc_times.append(dt)
                model_ms.append(sum(result.speed.values()))
            fps = ema.fps
            t_s = frame_idx / src_fps if is_file else time.perf_counter() - t_start

            counts[out.state] += 1
            if out.new_lock:
                acquisitions += 1
                if out.target.track_id not in locked_ids:
                    locked_ids.append(out.target.track_id)

            if csv_writer:
                csv_writer.writerow(csv_row(frame_idx, t_s, fps, out))
            draw_overlay(frame, out, tracks, dets, names, fps, device_label, frame_idx, t_s, args.coast_frames)
            if args.save:
                if sink is None:
                    sink = VideoSink(args.save, src_fps, (w, h))
                sink.write(frame)
            if gif:
                gif.add(n, frame)
            if show:
                cv2.imshow("drone-detect lock-on", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    interrupted = True
                    break

            n += 1
            frame_idx += 1
            if time.perf_counter() - t_log > 5:
                t_log = time.perf_counter()
                total = f"/{n_total}" if n_total else ""
                tgt = f" #{out.target.track_id}" if out.target is not None else ""
                print(f"[demo] frame {frame_idx}{total}  FPS {fps or 0:.1f}  {out.state.value}{tgt}")
    except KeyboardInterrupt:
        interrupted = True
    finally:
        wall = time.perf_counter() - t_start
        cap.release()
        if sink is not None:
            sink.close()
        if csv_file:
            csv_file.close()
        if show:
            cv2.destroyAllWindows()

    gif_txt = gif.save() if gif else None
    print_summary(args, n, frame_idx, size, src_fps, hardware, proc_times, model_ms, wall, counts,
                  acquisitions, locked_ids, gif_txt, interrupted)  # fmt: skip


def _mb(p: Path) -> str:
    return f"{p.stat().st_size / 1e6:.1f} Mo" if p.exists() else "absent"


def print_summary(args, n, frame_idx, size, src_fps, hardware, proc_times, model_ms, wall, counts,
                  acquisitions, locked_ids, gif_txt, interrupted) -> None:  # fmt: skip
    print("\n=== Résumé démo lock-on ===")
    if n == 0:
        print("Aucune frame lue : vérifier --source / --start-frame.")
        return
    dims = f"{size[0]}x{size[1]}" if size else "?"
    print(f"Source        : {args.source} ({dims}, {src_fps:.2f} fps){' — interrompu' if interrupted else ''}")
    print(f"Modèle        : {args.weights}, tracker {args.tracker}, imgsz {args.imgsz}, device {args.device or 'auto'}")
    print(f"Matériel      : {hardware}")
    print(f"Frames        : {n} traitées ({frame_idx - n} → {frame_idx - 1})")
    if proc_times:
        print(f"FPS pipeline  : {len(proc_times) / sum(proc_times):.1f} en moyenne "
              f"(lecture + inférence + suivi + verrouillage, hors 1re frame)")  # fmt: skip
        print(f"Latence modèle: {np.mean(model_ms):.1f} ms/frame (pré-traitement + inférence + post-traitement)")
    print(f"Débit global  : {n / wall:.1f} FPS (overlay et écritures compris, 1re frame comprise)")
    pct = " | ".join(f"{st.value} {100 * counts[st] / n:.1f} %" for st in LockState)
    ids = ", ".join(map(str, locked_ids)) or "aucun"
    print(f"Verrouillage  : {pct} ; {acquisitions} acquisition(s) ; IDs verrouillés : {ids}")
    if args.save:
        print(f"Vidéo         : {args.save} ({_mb(args.save)})")
    if args.csv:
        print(f"CSV           : {args.csv} ({n} lignes)")
    if gif_txt:
        print(f"GIF           : {gif_txt}")


if __name__ == "__main__":
    main()
