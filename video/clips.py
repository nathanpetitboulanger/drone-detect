"""Clips de démo : vidéo brute + pistes enregistrées par ``video/track_dump.py``.

``TrackClip.overlay(i, w, h)`` redessine l'overlay **exact** de la démo
(``scripts/demo.py::draw_overlay``) sur la frame ``i`` agrandie à ``w``×``h`` : mêmes
données (pistes, verrou, FPS mesuré), même fonction de dessin, mais à la résolution de la
vidéo de présentation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import demo  # noqa: E402
from lockon import AimError, LockOutput, LockState, Track  # noqa: E402

from vkit import VideoReader  # noqa: E402

TRACKS = ROOT / "outputs" / "video" / "tracks"


def tracks_json(stem: str, tag: str) -> Path:
    """Chemin du JSON de pistes ``outputs/video/tracks/<stem>__<tag>.json`` (écrit par ``track_dump.py``)."""
    p = TRACKS / f"{stem}__{tag}.json"
    if not p.exists():
        raise SystemExit(f"pistes absentes : {p} — lancer video/track_dump.py (voir video/STORYBOARD.md)")
    return p


class TrackClip:
    def __init__(self, stem: str, tag: str = "r2"):
        self.stem = stem
        self.video = ROOT / "data" / "demo" / f"{stem}.mp4"
        self.json = tracks_json(stem, tag)
        d = json.loads(self.json.read_text())
        self.meta, self.frames = d["meta"], d["frames"]
        self.by_i = {f["i"]: f for f in self.frames}
        self.reader = VideoReader(self.video)
        self.w, self.h = self.meta["width"], self.meta["height"]
        self.fps = self.meta["src_fps"] or 30.0
        self.names = {0: "drone"}

    def __len__(self) -> int:
        return len(self.frames)

    def f(self, i: int) -> dict:
        i = max(self.frames[0]["i"], min(i, self.frames[-1]["i"]))
        return self.by_i[i]

    def raw(self, i: int) -> np.ndarray:
        return self.reader.get(i)

    def lock_output(self, i: int, kx: float = 1.0, ky: float | None = None):
        """(LockOutput, pistes, détections) de la frame ``i``, coordonnées x multipliées par ``kx``, y par ``ky``."""
        fr = self.f(i)
        ky = kx if ky is None else ky

        def tr(t):
            cls = int(t[6]) if len(t) > 6 else 0
            return Track(int(t[0]), t[1] * kx, t[2] * ky, t[3] * kx, t[4] * ky, float(t[5]), cls)

        tracks = [tr(t) for t in fr["tracks"]]
        dets = [(a * kx, b * ky, c * kx, e * ky, cf, cl) for a, b, c, e, cf, cl in fr["dets"]]
        target = tr(fr["tgt"]) if fr["tgt"] else None
        aim = None
        if fr["aim"]:
            dx, dy, area = fr["aim"]
            aim = AimError(dx, dy, area, float(np.sqrt(area)))
        out = LockOutput(LockState(fr["state"]), target, aim, fr["lost"], fr["n"], fr["new"])
        return out, tracks, dets

    def overlay(self, i: int, w: int | None = None, h: int | None = None) -> Image.Image:
        """Frame ``i`` agrandie à ``w``×``h`` avec l'overlay de ``demo.draw_overlay`` (RGB PIL)."""
        w, h = w or self.w, h or self.h
        img = self.raw(i)
        kx, ky = w / self.w, h / self.h
        if (w, h) != (self.w, self.h):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4 if kx > 1 else cv2.INTER_AREA)
        else:
            img = img.copy()
        out, tracks, dets = self.lock_output(i, kx, ky)
        fr = self.f(i)
        demo.draw_overlay(img, out, tracks, dets, self.names, fr["fps"], self.meta["device_label"], fr["i"],
                          fr["i"] / self.fps, self.meta["coast_frames"])  # fmt: skip
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # --- Analyse des segments ------------------------------------------------------

    def states(self, a: int, b: int) -> list[str]:
        return [self.f(i)["state"] for i in range(a, b)]

    def pick_segment(self, n: int, lead: int, lo: int = 0, hi: int | None = None) -> int:
        """Début d'un segment de ``n`` frames où une acquisition a lieu ``lead`` frames après le début.

        Critère : part de frames LOCKED dans le segment (déterministe ; à égalité, le plus tôt).
        """
        hi = self.frames[-1]["i"] if hi is None else hi
        best, best_s = None, -1.0
        for fr in self.frames:
            if not fr["new"]:
                continue
            s0 = fr["i"] - lead
            if s0 < lo or s0 + n > hi + 1:
                continue
            score = sum(st == "LOCKED" for st in self.states(s0, s0 + n)) / n
            if score > best_s + 1e-9:
                best, best_s = s0, score
        if best is None:
            best = max(lo, min(hi + 1 - n, lo))
        return best

    def centers(self, a: int, b: int, alpha: float = 0.3) -> list[tuple[float, float, float]]:
        """Centre lissé (EMA) et taille de la cible pour la loupe, frames ``a``…``b - 1``.

        Hors verrou, garde la dernière position (ou la 1re position connue du segment)."""
        raw = []
        for i in range(a, b):
            t = self.f(i)["tgt"]
            raw.append(None if t is None else ((t[1] + t[3]) / 2, (t[2] + t[4]) / 2, max(t[3] - t[1], t[4] - t[2])))
        first = next((r for r in raw if r is not None), None)
        if first is None:
            first = (self.w / 2, self.h / 2, 40.0)
        out, cur = [], first
        for r in raw:
            if r is not None:
                cur = tuple(c + alpha * (v - c) for c, v in zip(cur, r))
            out.append(cur)
        return out
