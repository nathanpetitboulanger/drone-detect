"""Rendu de la vidéo de présentation : scènes → segments → master → MP4 final ≤ 10 Mo + affiche.

Étapes (toutes idempotentes) :
  1. chaque scène de ``video/scenes.py`` est rendue image par image (Pillow/OpenCV) et
     encodée en H.264 quasi sans perte dans ``outputs/video/segments/<scène>.mp4`` ;
  2. les segments sont concaténés dans ``outputs/video/master.mp4`` ;
  3. encodage final H.264 ``yuv420p`` en 2 passes avec débit calculé pour tenir sous
     ``--max-mb`` (10 Mo, limite d'upload vidéo GitHub), ``+faststart`` →
     ``docs/assets/presentation.mp4`` ;
  4. affiche ``docs/assets/presentation_poster.jpg`` (une image de la scène d'accroche) ;
  5. images de contrôle de chaque scène dans ``outputs/video/review/``.

Prérequis : polices (``video/fetch_fonts.py``), pistes de démo (``video/track_dump.py``).

Usage (depuis la racine du dépôt) :
    .venv/bin/python video/render.py                       # tout (rendu, encodage, affiche, contrôle)
    .venv/bin/python video/render.py --scenes hook demo19  # ne re-rendre que ces scènes, puis tout ré-assembler
    .venv/bin/python video/render.py --preview 1.5 6       # aperçus PNG de chaque scène à t = 1,5 s et 6 s
    .venv/bin/python video/render.py --list                # scènes, durées, minutage
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

OUT = ROOT / "outputs" / "video"
SEG = OUT / "segments"
FINAL = ROOT / "docs" / "assets" / "presentation.mp4"
POSTER = ROOT / "docs" / "assets" / "presentation_poster.jpg"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", nargs="+", default=None, help="scènes à (re)rendre (défaut : celles sans segment)")
    p.add_argument("--all", action="store_true", help="re-rendre toutes les scènes")
    p.add_argument("--preview", nargs="+", type=float, default=None, metavar="T",
                   help="seulement des aperçus PNG à ces instants (s) de chaque scène sélectionnée")  # fmt: skip
    p.add_argument("--list", action="store_true", help="afficher le minutage et quitter")
    p.add_argument("--jobs", type=int, default=4, help="processus de rendu en parallèle")
    p.add_argument("--max-mb", type=float, default=9.6, help="taille visée du MP4 final (Mo, 1 Mo = 10^6 o)")
    p.add_argument("--no-encode", action="store_true", help="s'arrêter au master")
    return p.parse_args()


def scene_list():
    import scenes

    return scenes.all_scenes()


def fade_k(sc, t: float) -> float:
    from vkit import ease_in_out

    k = 1.0
    if sc.fade_in > 0:
        k = min(k, ease_in_out(t / sc.fade_in))
    if sc.fade_out > 0:
        k = min(k, ease_in_out((sc.dur - t) / sc.fade_out))
    return k


def render_frame(sc, t: float):
    from vkit import fade_frame

    return fade_frame(sc.draw(t), fade_k(sc, t))


def render_scene(name: str) -> tuple[str, float, int]:
    """Rend une scène dans son segment (processus séparé)."""
    from vkit import FPS, H, W

    sc = next(s for s in scene_list() if s.name == name)
    sc.setup()
    n = int(round(sc.dur * FPS))
    SEG.mkdir(parents=True, exist_ok=True)
    tmp = SEG / f"{name}.tmp.mp4"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "10", "-pix_fmt", "yuv420p",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(tmp)]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.perf_counter()
    for f in range(n):
        img = render_frame(sc, (f + 0.5) / FPS)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg a échoué pour la scène {name}")
    tmp.replace(SEG / f"{name}.mp4")
    return name, time.perf_counter() - t0, n


def preview(names: list[str], times: list[float]) -> None:
    pdir = OUT / "preview"
    pdir.mkdir(parents=True, exist_ok=True)
    for sc in scene_list():
        if sc.name not in names:
            continue
        sc.setup()
        for t in times:
            if t > sc.dur:
                continue
            p = pdir / f"{sc.name}_{t:05.2f}.png"
            render_frame(sc, t).save(p)
            print(p.relative_to(ROOT))


def concat(names: list[str]) -> Path:
    lst = OUT / "segments.txt"
    lst.write_text("".join(f"file '{SEG / n}.mp4'\n" for n in names))
    master = OUT / "master.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(master)], check=True)  # fmt: skip
    return master


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration,size,bit_rate:stream=codec_name,profile,pix_fmt,width,height,r_frame_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)  # fmt: skip
    return json.loads(out.stdout)


def encode(master: Path, max_mb: float) -> None:
    """2 passes x264 ; si le fichier dépasse, on recommence avec un débit réduit."""
    dur = float(probe(master)["format"]["duration"])
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    plog = OUT / "x264pass"
    budget = max_mb * 1e6 * 8 / dur / 1000 * 0.985  # kbit/s, marge conteneur
    for attempt in range(4):
        kbps = int(budget)
        base = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(master), "-c:v", "libx264", "-preset", "veryslow",
                "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 2.5)}k", "-bufsize", f"{int(kbps * 4)}k",
                "-x264-params", "aq-mode=3:aq-strength=0.9:deblock=-1,-1", "-pix_fmt", "yuv420p",
                "-profile:v", "high", "-level", "4.1", "-g", "120", "-passlogfile", str(plog), "-an"]  # fmt: skip
        subprocess.run(base + ["-pass", "1", "-f", "mp4", os.devnull], check=True)
        subprocess.run(base + ["-pass", "2", "-movflags", "+faststart", str(FINAL)], check=True)
        mb = FINAL.stat().st_size / 1e6
        print(f"[render] encodage {kbps} kbit/s → {mb:.2f} Mo")
        if mb <= max_mb:
            break
        budget *= max_mb / mb * 0.97
    for f in OUT.glob("x264pass*"):
        f.unlink()


def poster(t: float = 7.2) -> None:
    """Affiche : une image de l'accroche (titre lisible), JPEG."""
    import scenes

    sc = scenes.Hook()
    sc.setup()
    img = render_frame(sc, t)
    img.save(POSTER, quality=86, optimize=True, progressive=True)
    print(f"[render] affiche {POSTER.relative_to(ROOT)} ({POSTER.stat().st_size / 1e3:.0f} ko)")


def review(names_durs: list[tuple[str, float]]) -> None:
    """Images de contrôle extraites du MP4 final (2 par scène) + planche."""
    rdir = OUT / "review"
    rdir.mkdir(parents=True, exist_ok=True)
    t0 = 0.0
    for name, dur in names_durs:
        for frac in (0.45, 0.9):
            ts = t0 + dur * frac
            p = rdir / f"{ts:06.2f}_{name}.jpg"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{ts:.3f}", "-i", str(FINAL), "-frames:v",
                            "1", "-q:v", "3", str(p)], check=True)  # fmt: skip
        t0 += dur
    print(f"[render] images de contrôle : {rdir.relative_to(ROOT)}/")


def main() -> None:
    args = parse_args()
    if not (ROOT / "video" / "fonts" / "Inter-Variable.ttf").exists():
        import fetch_fonts

        fetch_fonts.main([])
    scs = scene_list()
    names = [s.name for s in scs]
    if args.list:
        t = 0.0
        for s in scs:
            print(f"{t:6.1f}–{t + s.dur:6.1f} s  {s.name:10s} ({s.dur:.1f} s)")
            t += s.dur
        print(f"total {t:.1f} s")
        return
    sel = args.scenes or names
    if args.preview is not None:
        preview(sel, args.preview)
        return
    todo = names if args.all else (args.scenes or [n for n in names if not (SEG / f"{n}.mp4").exists()])
    unknown = set(todo) - set(names)
    if unknown:
        raise SystemExit(f"scènes inconnues : {sorted(unknown)} ; disponibles : {names}")
    if todo:
        t0 = time.perf_counter()
        with Pool(min(args.jobs, len(todo))) as pool:
            for name, dt, n in pool.imap_unordered(render_scene, todo):
                print(f"[render] {name:10s} {n} images en {dt:.0f} s")
        print(f"[render] rendu : {time.perf_counter() - t0:.0f} s")
    master = concat(names)
    if args.no_encode:
        return
    encode(master, args.max_mb)
    poster()
    info = probe(FINAL)
    st, fm = info["streams"][0], info["format"]
    print(f"[render] {FINAL.relative_to(ROOT)} : {st['codec_name']} {st['profile']} {st['pix_fmt']} "
          f"{st['width']}x{st['height']} @ {st['r_frame_rate']}, {float(fm['duration']):.1f} s, "
          f"{int(fm['size']) / 1e6:.2f} Mo")  # fmt: skip
    review([(s.name, s.dur) for s in scs])


if __name__ == "__main__":
    main()
