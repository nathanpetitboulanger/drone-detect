"""Reconstruit en mp4 (H.264) les séquences de test réservées à la démo, à partir des données brutes.

Les séquences démo sont celles marquées `demo=1` dans <yolo-dir>/splits.csv (liste DEMO_SEQUENCES de
split.py) ; le script vérifie qu'elles sont bien en test. Toutes les frames d'origine sont utilisées
(pas le sous-échantillonnage du jeu YOLO) :
  - halmstad  : la vidéo d'origine (déjà en H.264, 30 fps) est copiée telle quelle ;
  - dut_track : les JPEG de la séquence sont encodés à `--fps` (FPS d'origine non documenté : 30 par défaut).

Sortie : data/demo/<source>__<sequence>.mp4 (idempotent : ignoré si déjà présent, sauf --force).
Mode libre : `--frames-dir DIR --out X.mp4` encode n'importe quel dossier d'images triées.

Usage :
    .venv/bin/python scripts/data/make_video.py
    .venv/bin/python scripts/data/make_video.py --frames-dir data/raw/dut_track/Anti-UAV-Tracking-V0/video07 --out data/demo/x.mp4
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

from common import DEMO, RAW, add_yolo_dir_arg, layout_from


def encode_frames(frames_dir: Path, out: Path, fps: float, pattern: str = "*.jpg") -> None:
    """Encode les images triées d'un dossier en mp4 H.264 (yuv420p, lisible partout)."""
    frames = sorted(frames_dir.glob(pattern))
    if not frames:
        raise SystemExit(f"aucune image {pattern} dans {frames_dir}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{fps:g}",
           "-pattern_type", "glob", "-i", str(frames_dir / pattern),
           "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
           # dimensions paires exigées par yuv420p
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(out)]
    subprocess.run(cmd, check=True)
    print(f"  {len(frames)} frames @ {fps:g} fps → {out}")


def build_demo(source: str, sequence: str, out: Path, fps: float) -> None:
    if source == "halmstad":
        src = RAW / "halmstad" / "Data" / "Video_V" / f"{sequence}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        print(f"  copie {src.name} (H.264 d'origine, 30 fps) → {out}")
    elif source == "dut_track":
        encode_frames(RAW / "dut_track" / "Anti-UAV-Tracking-V0" / sequence, out, fps)
    else:
        raise SystemExit(f"source {source} : pas de vidéo reconstructible")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, default=30.0, help="FPS pour les séquences d'images (défaut 30)")
    parser.add_argument("--force", action="store_true", help="réécrit les mp4 existants")
    parser.add_argument("--frames-dir", type=Path, help="mode libre : dossier d'images à encoder")
    parser.add_argument("--out", type=Path, help="mode libre : mp4 de sortie")
    add_yolo_dir_arg(parser)
    args = parser.parse_args()

    if args.frames_dir:
        if not args.out:
            parser.error("--out requis avec --frames-dir")
        encode_frames(args.frames_dir, args.out, args.fps)
        return

    table = layout_from(args.yolo_dir).root / "splits.csv"
    with table.open() as f:
        demos = [r for r in csv.DictReader(f) if r["demo"] == "1"]
    if not demos:
        raise SystemExit(f"aucune séquence demo=1 dans {table}")
    for r in demos:
        if r["split"] != "test":
            raise SystemExit(f"{r['source']}/{r['sequence']} est en {r['split']}, pas en test : refus")
        out = DEMO / f"{r['source']}__{r['sequence']}.mp4"
        if out.exists() and not args.force:
            print(f"  {out} : déjà présent")
            continue
        build_demo(r["source"], r["sequence"], out, args.fps)


if __name__ == "__main__":
    main()
