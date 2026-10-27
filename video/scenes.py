"""Scènes de la vidéo de présentation (voir ``video/STORYBOARD.md``).

Chaque scène est une classe avec ``name``, ``dur`` (s) et ``draw(t) -> PIL.Image`` RGB
1920×1080 ; ``setup()`` charge les ressources (appelé dans le processus de rendu).
Tous les chiffres viennent de ``facts.py`` ; les séquences de démo, de ``clips.py``.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import facts as F
from clips import TrackClip
from vkit import (
    C, FPS, H, MX, MY, STATE_RGB, W, VideoReader, appear, blit, bgr_to_pil, circle, clamp, cover, cv_circle,
    cv_polyline, ease_in_out, ease_out, fmt_fr, lerp, load_image, mix, new_frame, para,
    paste_rounded, pill, prog, rrect, shadow, text, vector,
)  # fmt: skip

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


def ff(name: str, nd: int = 2) -> str:
    """Valeur de ``facts`` formatée à la française."""
    return fmt_fr(F.v(name), nd)


# --- Éléments communs -----------------------------------------------------------------


def header(fr, t, kicker: str, title: str, t0: float = 0.0, end: float | None = None, size: int = 64):
    """Sur-titre (mono, rouge) + titre, avec une entrée glissée."""
    a1, a2 = appear(t, t0, 0.5, end), appear(t, t0 + 0.15, 0.55, end)
    d1 = (1 - ease_out(prog(t, t0, 0.6))) * 16
    d2 = (1 - ease_out(prog(t, t0 + 0.15, 0.6))) * 16
    blit(fr, rrect(12, 12, 2, fill=C["red"]), MX, MY + 15 + d1, a1)
    blit(fr, text(kicker, 40, C["red"], 600, mono=True, track=0.04), MX + 28, MY + d1, a1)
    blit(fr, text(title, size, C["text"], 700), MX, MY + 58 + d2, a2)


def panel(fr, x, y, w, h, a=1.0, fill=C["panel"], r=18, outline=C["line"]):
    blit(fr, shadow(w, h, r), x, y, a * 0.9)
    blit(fr, rrect(w, h, r, fill=fill, outline=outline, width=2), x, y, a)


def slide(t, t0, dist=24.0, dur=0.6) -> float:
    return (1 - ease_out(prog(t, t0, dur))) * dist


def arrow(fr, x0, y0, x1, y1, color, p=1.0, a=1.0, width=4, head=16, dash=0):
    """Flèche anticrénelée de (x0, y0) vers (x1, y1), tracée jusqu'à la fraction ``p``."""
    if p <= 0 or a <= 0:
        return
    xe, ye = lerp(x0, x1, p), lerp(y0, y1, p)
    pad = head + width + 4
    bx0, by0 = int(min(x0, xe) - pad), int(min(y0, ye) - pad)
    bw, bh = int(abs(xe - x0) + 2 * pad), int(abs(ye - y0) + 2 * pad)

    def draw(d, k):
        sx, sy, ex, ey = (x0 - bx0) * k, (y0 - by0) * k, (xe - bx0) * k, (ye - by0) * k
        ang = math.atan2(ey - sy, ex - sx)
        hl = head * k
        bx, by = ex - math.cos(ang) * hl * 0.9, ey - math.sin(ang) * hl * 0.9
        if dash:
            L = math.hypot(bx - sx, by - sy)
            n = max(1, int(L / (2 * dash * k)))
            for j in range(n + 1):
                s0, s1 = j * 2 * dash * k, min(L, j * 2 * dash * k + dash * k)
                if s0 >= L:
                    break
                d.line([(sx + (bx - sx) * s0 / L, sy + (by - sy) * s0 / L), (sx + (bx - sx) * s1 / L,
                        sy + (by - sy) * s1 / L)], fill=(*color, 255), width=int(width * k))  # fmt: skip
        else:
            d.line([(sx, sy), (bx, by)], fill=(*color, 255), width=int(width * k))
        pts = [(ex, ey), (ex - math.cos(ang - 0.45) * hl, ey - math.sin(ang - 0.45) * hl),
               (ex - math.cos(ang + 0.45) * hl, ey - math.sin(ang + 0.45) * hl)]  # fmt: skip
        d.polygon(pts, fill=(*color, 255))

    blit(fr, vector(bw, bh, draw), bx0, by0, a)


def brackets_img(w, h, color, thick=4, frac=0.28) -> Image.Image:
    """Coins de verrouillage (style overlay) dans un calque ``w``×``h``."""
    w, h = int(w), int(h)

    def draw(d, k):
        L = frac * min(w, h) * k
        x0, y0, x1, y1 = thick * k / 2, thick * k / 2, w * k - thick * k / 2, h * k - thick * k / 2
        for x, y, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            d.line([(x, y + sy * L), (x, y), (x + sx * L, y)], fill=(*color, 255), width=int(thick * k), joint="curve")

    return vector(w, h, draw)


def reticle_img(r, color=C["text"], thick=3) -> Image.Image:
    s = 2 * r + 8

    def draw(d, k):
        c = s * k / 2
        rr = r * 0.55 * k
        d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=(*color, 255), width=int(thick * k))
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d.line([(c + sx * r * 0.25 * k, c + sy * r * 0.25 * k), (c + sx * r * k, c + sy * r * k)],
                   fill=(*color, 255), width=int(thick * k))  # fmt: skip

    return vector(s, s, draw)


def thumb_with_target(img: Image.Image, box, out_w, out_h, zoom_box: float = 6.0, min_crop: int = 160):
    """Recadrage centré sur une boîte (x1, y1, x2, y2) → (vignette, transform px→vignette)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * zoom_box
    cw = max(min_crop * out_w / out_h, side * out_w / out_h)
    ch = cw * out_h / out_w
    cw, ch = min(cw, img.width), min(ch, img.height)
    ox = clamp(cx - cw / 2, 0, img.width - cw)
    oy = clamp(cy - ch / 2, 0, img.height - ch)
    th = img.resize((out_w, out_h), Image.Resampling.LANCZOS, box=(ox, oy, ox + cw, oy + ch))
    s = out_w / cw
    return th, (lambda px, py: ((px - ox) * s, (py - oy) * s)), s


# --- Graphe dx / dy -------------------------------------------------------------------


class AimGraph:
    """Graphe dx, dy (sans unité, [-1, 1]) d'un segment, tracé progressivement + bande d'état."""

    def __init__(self, clip: TrackClip, a: int, n: int, w: int, h: int):
        self.w, self.h, self.n = w, h, n
        self.rows = [clip.f(a + j) for j in range(n)]
        self.pl, self.pr, self.pt, self.pb = 92, 34, 84, 58  # marges internes
        self.strip_h = 12

    def xy(self, j, v):
        gw, gh = self.w - self.pl - self.pr, self.h - self.pt - self.pb
        return self.pl + gw * j / max(1, self.n - 1), self.pt + gh * (1 - (v + 1) / 2)

    def draw(self, fr, x, y, j_now: int, a: float = 1.0):
        w, h = self.w, self.h
        bg = Image.new("RGB", (w, h), C["panel"])
        arr = np.asarray(bg).copy()
        gx0, gx1 = self.pl, w - self.pr
        for v in (-1, 0, 1):
            _, yy = self.xy(0, v)
            cv_polyline(arr, [(gx0, yy), (gx1, yy)], C["line"] if v == 0 else C["grid"], 2)
        # Bande d'état (SEARCHING / LOCKED / COASTING) sous le graphe.
        sy = h - self.pb + 22
        for j in range(min(j_now + 1, self.n)):
            x0, _ = self.xy(j, 0)
            x1, _ = self.xy(j + 1, 0) if j + 1 < self.n else (gx1, 0)
            col = STATE_RGB[self.rows[j]["state"]]
            cv2.rectangle(arr, (int(x0), sy), (int(math.ceil(x1)), sy + self.strip_h), col, -1)
        # Courbes : pleines en LOCKED, atténuées en COASTING (donnée périmée), coupées en SEARCHING.
        for k, key in ((0, "dx"), (1, "dy")):
            seg, seg_state = [], None
            for j in range(min(j_now + 1, self.n)):
                r = self.rows[j]
                st = r["state"] if r["aim"] else None
                if st != seg_state or st is None:
                    self._flush(arr, seg, seg_state, key)
                    seg = [seg[-1]] if (seg and st and seg_state) else []
                    seg_state = st
                if st:
                    seg.append(self.xy(j, r["aim"][k]))
            self._flush(arr, seg, seg_state, key)
        # Curseur temporel.
        cx, _ = self.xy(min(j_now, self.n - 1), 0)
        cv_polyline(arr, [(cx, self.pt - 8), (cx, h - self.pb + 4)], C["dim"], 2)
        r = self.rows[min(j_now, self.n - 1)]
        if r["aim"]:
            for k, key in ((0, "dx"), (1, "dy")):
                px, py = self.xy(min(j_now, self.n - 1), r["aim"][k])
                cv_circle(arr, px, py, 7, C[key])
        img = Image.fromarray(arr)
        # Étiquettes (≥ 40 px).
        for v, lab in ((1, "+1"), (0, "0"), (-1, "−1")):
            _, yy = self.xy(0, v)
            blit(img, text(lab, 40, C["dim"], 500, mono=True), self.pl - 18, yy, 1, "rm")
        xx = gx0 + 6
        for k, key in ((0, "dx"), (1, "dy")):
            val = f"{r['aim'][k]:+.2f}" if r["aim"] else "  --"
            sp_k, sp_v = text(key, 40, C[key], 700, mono=True), text(val, 40, C["text"], 500, mono=True)
            blit(img, sp_k, xx, 8)
            blit(img, sp_v, xx + sp_k.w + 14, 8)
            xx += sp_k.w + 14 + sp_v.w + 44
        paste_rounded(fr, img, x, y, 16, a)
        blit(fr, rrect(w, h, 16, outline=C["line"], width=2), x, y, a)

    @staticmethod
    def _flush(arr, seg, st, key):
        if st and len(seg) >= 2:
            col = C[key] if st == "LOCKED" else mix(C[key], C["panel"], 0.55)
            cv_polyline(arr, seg, col, 4 if st == "LOCKED" else 3)


STATE_DESC = {"SEARCHING": "en attente", "LOCKED": "cible suivie", "COASTING": "cible perdue"}


def state_legend(fr, rows: list[dict], j: int, x: float, y: float, a: float = 1.0) -> None:
    """Légende des 3 états ; l'état courant (frame ``j`` de ``rows``) est allumé, les autres atténués.

    L'intensité est la part de l'état sur les 4 dernières frames (transition douce, sans retard visible).
    """
    recent = [rows[q]["state"] for q in range(max(0, j - 3), j + 1)]
    for st, desc in STATE_DESC.items():
        on = sum(r == st for r in recent) / len(recent)
        col = STATE_RGB[st]
        sp = pill(st, 40, col, fill=mix(C["bg"], col, 0.16))
        blit(fr, sp, x, y, a * (0.28 + 0.72 * on))
        tx = text(desc, 40, mix(C["dim"], C["text"], on), 500)
        blit(fr, tx, x + sp.w + 16, y + sp.h / 2, a, "lm")
        x += sp.w + 16 + tx.w + 56


# --- Scènes ---------------------------------------------------------------------------


class Scene:
    name = "scene"
    dur = 5.0
    fade_in = 0.35
    fade_out = 0.35

    def setup(self) -> None:
        pass

    def draw(self, t: float) -> Image.Image:  # pragma: no cover - abstrait
        raise NotImplementedError


class Hook(Scene):
    """Accroche : vrai clip, le drone se fait verrouiller ; titre du projet."""

    name, dur, fade_in = "hook", 8.0, 0.25
    stem = "dut_track__video18"

    def setup(self):
        self.clip = TrackClip(self.stem)
        self.n = int(self.dur * FPS)
        self.a = self.clip.pick_segment(self.n, lead=int(1.2 * FPS))
        hb = 560
        u = np.clip(np.linspace(0, 1, hb) / 0.6, 0, 1)
        alpha = (240 * (u * u * (3 - 2 * u)))[:, None].repeat(W, 1)
        rgb = np.broadcast_to(np.array(C["bg"], np.uint8), (hb, W, 3))
        self.band = Image.fromarray(np.dstack([rgb, alpha.astype(np.uint8)]), "RGBA")

    def draw(self, t):
        i = self.a + min(self.n - 1, int(t * FPS))
        fr = self.clip.overlay(i, W, H)
        ta = 2.6
        tag = pill(
            "Séquence de test, jamais vue à l'entraînement",
            40,
            C["text"],
            fill=C["bg"],
            weight=500,
            mono=False,
            outline=False,
        )
        blit(fr, tag, W / 2, 36, appear(t, 0.3, 0.4, ta + 0.2), "mt")
        blit(fr, self.band, 0, H - self.band.height, appear(t, ta, 0.8))
        y0 = H - MY - 250
        blit(fr, text("GUIDAGE FINAL PAR VISION · ANTI-DRONE", 40, C["red"], 600, mono=True, track=0.04),
             MX, y0 + slide(t, ta + 0.1), appear(t, ta + 0.1, 0.5))  # fmt: skip
        blit(fr, text("drone-detect", 112, C["text"], 760), MX - 4, y0 + 52 + slide(t, ta + 0.25),
             appear(t, ta + 0.25, 0.6))  # fmt: skip
        blit(fr, text("Détecter, suivre et verrouiller un drone par vision", 48, C["text"], 450), MX,
             y0 + 192 + slide(t, ta + 0.5), appear(t, ta + 0.5, 0.6))  # fmt: skip
        return fr


class Context(Scene):
    """Le problème et ses difficultés, illustrées par de vraies images."""

    name, dur = "context", 10.0

    def setup(self):
        cw, ch = 408, 300
        self.cw, self.ch = cw, ch
        # 1. Cible minuscule : image entière Halmstad réduite, drone entouré.
        c35 = TrackClip("halmstad__V_DRONE_035")
        f = c35.f(30)
        img = bgr_to_pil(c35.raw(30))
        tgt = f["tgt"] or (f["tracks"][0] if f["tracks"] else None)
        im1 = cover(img, cw, ch)
        if tgt:
            s = max(cw / img.width, ch / img.height)
            ox, oy = (img.width * s - cw) / 2, (img.height * s - ch) / 2
            cx, cy = (tgt[1] + tgt[3]) / 2 * s - ox, (tgt[2] + tgt[4]) / 2 * s - oy
            ring = circle(26, outline=C["yellow"], width=3)
            blit(im1, ring, cx, cy, 1, "mm")
        # 2. Oiseaux.
        im2 = cover(bgr_to_pil(VideoReader(DEMO / "halmstad__V_BIRD_020.mp4").get(10)), cw, ch, 0.55, 0.78, 1.5)
        # 3. Fond encombré (DUT video19).
        im3 = cover(bgr_to_pil(VideoReader(DEMO / "dut_track__video19.mp4").get(600)), cw, ch, 0.45, 0.55, 1.25)
        # 4. Temps réel : schéma « une image toutes les 33 ms ».
        im4 = Image.new("RGB", (cw, ch), C["panel2"])
        for j in range(5):
            x = 34 + j * 70
            blit(im4, rrect(56, 96, 8, fill=mix(C["panel2"], C["text"], 0.12 + 0.05 * j), outline=C["line"]), x, 58)
        blit(im4, text("33 ms", 48, C["text"], 700, mono=True), cw / 2, 196, 1, "mm")
        self.cards = [
            (im1, "Cibles minuscules", f"taille médiane : {ff('test_median_px', 0)}\u00a0px dans le test"),
            (im2, "Oiseaux, aéronefs", "des leurres naturels"),
            (im3, "Fonds encombrés", "bâtiments, arbres, nuages"),
            (im4, "Temps réel embarqué", "30 images/s, sur un module embarqué"),
        ]

    def draw(self, t):
        fr = new_frame()
        header(fr, t, "LE PROBLÈME", "Guider un intercepteur jusqu'au drone, par la vision")
        blit(fr, para("La caméra doit trouver la cible, la garder et mesurer son écart au centre.",
                      44, 1500, C["muted"], 450), MX, 236 + slide(t, 0.5), appear(t, 0.5, 0.6))  # fmt: skip
        gap = (W - 2 * MX - 4 * self.cw) / 3
        blocks = [(para(ti, 44, self.cw - 44, C["text"], 650), para(su, 40, self.cw - 44, C["muted"], 450, lh=1.2))
                  for _, ti, su in self.cards]  # fmt: skip
        body_h = max(a.h + 14 + b.h for a, b in blocks)
        for k, ((img, _, _), (sp_t, sp_s)) in enumerate(zip(self.cards, blocks)):
            t0 = 1.4 + 0.45 * k
            a = appear(t, t0, 0.6)
            x = MX + k * (self.cw + gap)
            y = 350 + slide(t, t0, 30)
            panel(fr, x, y, self.cw, self.ch + body_h + 60, a)
            paste_rounded(fr, img, x, y, 18, a)
            blit(fr, rrect(self.cw, 4, 0, fill=C["red"]), x, y + self.ch - 2, a)
            blit(fr, sp_t, x + 22, y + self.ch + 26, a)
            blit(fr, sp_s, x + 22, y + self.ch + 26 + sp_t.h + 14, a)
        return fr


class Pipeline(Scene):
    """Chaîne caméra → YOLO11n → ByteTrack → verrou → (dx, dy) → [guidage hors périmètre]."""

    name, dur = "pipeline", 10.0
    stem = "dut_track__video19"

    def setup(self):
        clip = TrackClip(self.stem)
        # Frame verrouillée avec la plus longue histoire de suivi (pour la traînée).
        # Frame verrouillée la plus proche du toit (cible la plus basse) : décor visible dans les vignettes,
        # avec au moins 25 frames d'historique de la même piste pour la traînée.
        cands = [f for f in clip.frames if f["state"] == "LOCKED" and f["i"] > 40
                 and all(clip.f(f["i"] - q)["tgt"] and clip.f(f["i"] - q)["tgt"][0] == f["tgt"][0] for q in (1, 25))]  # fmt: skip
        f = max(cands, key=lambda g: (g["tgt"][2] + g["tgt"][4]) / 2)
        i = f["i"]
        raw = bgr_to_pil(clip.raw(i))
        box = f["tgt"][1:5]
        tw, th = 280, 180
        self.tw, self.th = tw, th
        t1, tf, s = thumb_with_target(raw, box, tw, th, zoom_box=7.0)
        # 2. Détections (boîtes brutes de la frame, blanches).
        t2 = t1.copy()
        d2 = ImageDraw.Draw(t2)
        for tr in f["tracks"]:
            (a, b), (c, e) = tf(tr[1], tr[2]), tf(tr[3], tr[4])
            d2.rectangle([a, b, c, e], outline=C["text"], width=2)
        for dt in f["dets"]:
            (a, b), (c, e) = tf(dt[0], dt[1]), tf(dt[2], dt[3])
            d2.rectangle([a, b, c, e], outline=C["muted"], width=2)
        # 3. Suivi : traînée des 25 dernières positions de la piste cible.
        t3 = t2.copy()
        arr = np.asarray(t3).copy()
        pts = []
        for j in range(i - 25, i + 1):
            g = clip.f(j)
            for tr in g["tracks"]:
                if tr[0] == f["tgt"][0]:
                    pts.append(tf((tr[1] + tr[3]) / 2, (tr[2] + tr[4]) / 2))
        cv_polyline(arr, pts, C["cyan"], 3)
        for tr in f["tracks"]:
            (a, b), (c, e) = tf(tr[1], tr[2]), tf(tr[3], tr[4])
            cv2.rectangle(arr, (int(a), int(b)), (int(c), int(e)), C["cyan"], 2, cv2.LINE_AA)
        t3 = Image.fromarray(arr)
        # 4. Verrou : overlay exact de la démo, recadré.
        ov = clip.overlay(i)
        t4 = thumb_with_target(ov, box, tw, th, zoom_box=7.0)[0]
        # 5. Erreur de visée : valeurs de cette frame.
        t5 = Image.new("RGB", (tw, th), C["panel2"])
        dx, dy = f["aim"][0], f["aim"][1]
        blit(t5, text(f"dx {dx:+.3f}", 42, C["dx"], 650, mono=True), tw / 2, th / 2 - 30, 1, "mm")
        blit(t5, text(f"dy {dy:+.3f}", 42, C["dy"], 650, mono=True), tw / 2, th / 2 + 32, 1, "mm")
        self.nodes = [
            ("Caméra", "flux vidéo", t1),
            ("YOLO11n", "détection", t2),
            ("ByteTrack", "suivi, ID", t3),
            ("Verrouillage", "3 états", t4),
            ("dx, dy", "erreur de visée", t5),
        ]

    def draw(self, t):
        fr = new_frame()
        header(fr, t, "LA CHAÎNE", "De l'image à l'erreur de visée, à chaque image")
        n = len(self.nodes)
        nw, nh, pad = 304, 140, 12
        gap = (W - 2 * MX - n * nw) / (n - 1)
        y = 270
        node_h = nh + self.th + pad
        for k, (title, sub, th) in enumerate(self.nodes):
            t0 = 0.7 + 0.5 * k
            a = appear(t, t0, 0.5)
            x = MX + k * (nw + gap)
            dy = slide(t, t0, 20)
            hi = k == n - 1
            panel(fr, x, y + dy, nw, node_h, a, fill=C["panel"], outline=C["red"] if hi else C["line"])
            blit(fr, text(title, 44, C["text"], 700), x + nw / 2, y + 24 + dy, a, "mt")
            blit(fr, text(sub, 40, C["muted"], 450), x + nw / 2, y + 80 + dy, a, "mt")
            paste_rounded(fr, th, x + pad, y + nh + dy, 10, a)
            if k < n - 1:
                p = ease_in_out(prog(t, t0 + 0.3, 0.45))
                # Après l'apparition, un « flux » rouge parcourt les flèches en boucle.
                ph = ((t - 3.8) / 2.2 - k / (n - 1)) % 1.0 if t > 3.8 else 1.0
                col = mix(C["muted"], C["red"], math.exp(-(((ph - 0.08) / 0.06) ** 2)))
                ya = y + node_h / 2
                arrow(fr, x + nw + 6, ya, x + nw + gap - 6, ya, col, p, 1.0, width=4, head=14)
        # Hors périmètre : contrôleur de guidage (pointillés) sous le dernier nœud.
        t6 = 3.6
        a6 = appear(t, t6, 0.6)
        x5 = MX + (n - 1) * (nw + gap)
        yb = y + node_h
        y6 = yb + 130
        arrow(
            fr, x5 + nw / 2, yb + 10, x5 + nw / 2, y6 - 10, C["dim"], ease_in_out(prog(t, t6, 0.5)), a6, 4, 14, dash=10
        )
        blit(fr, rrect(nw, nh, 18, fill=C["bg"], outline=C["dim"], width=3, dash=12), x5, y6, a6)
        blit(fr, text("Guidage", 44, C["muted"], 700), x5 + nw / 2, y6 + 24, a6, "mt")
        blit(fr, text("hors périmètre", 40, C["dim"], 500), x5 + nw / 2, y6 + 80, a6, "mt")
        tx = appear(t, t6 + 0.5, 0.6)
        l1 = text("Aucun code de contrôle de vol ni de charge utile.", 44, C["text"], 650)
        l2 = para("Le dépôt s'arrête à l'erreur de visée : l'entrée d'un contrôleur de guidage.", 44, 1100,
                  C["muted"], 450)  # fmt: skip
        xr = x5 - 56
        blit(fr, l1, xr, y6 + nh / 2 - 8, tx, "rb")
        blit(fr, l2, xr - l1.w + l2.w if l2.w < l1.w else xr, y6 + nh / 2 + 8, tx, "rt")
        return fr


class Lock(Scene):
    """Machine à états du verrou + convention de l'erreur de visée."""

    name, dur = "lock", 10.0
    stem = "dut_track__video19"

    def setup(self):
        self.d = F.demo_defaults()
        clip = TrackClip(self.stem)
        # Frame où dx et dy sont tous deux nets (vecteur lisible), cible loin des bords.
        cands = [f for f in clip.frames if f["state"] == "LOCKED" and f["aim"] and max(map(abs, f["aim"][:2])) < 0.7]
        f = max(cands, key=lambda g: min(abs(g["aim"][0]), abs(g["aim"][1])))
        self.f = f
        self.iw, self.ih = 760, 428
        self.img = bgr_to_pil(clip.raw(f["i"])).resize((self.iw, self.ih), Image.Resampling.LANCZOS)
        self.sw, self.sh = clip.w, clip.h

    def draw(self, t):
        fr = new_frame()
        header(fr, t, "VERROUILLAGE", "Une cible, gardée tant que le suivi tient")
        d = self.d
        # Machine à états (colonne gauche).
        x0, y0 = MX, 300
        nodes = {"SEARCHING": (x0 + 20, y0), "LOCKED": (x0 + 20, y0 + 250), "COASTING": (x0 + 20, y0 + 500)}
        for k, (st, (x, y)) in enumerate(nodes.items()):
            a = appear(t, 0.6 + 0.35 * k, 0.5)
            blit(fr, pill(st, 44, STATE_RGB[st], fill=mix(C["bg"], STATE_RGB[st], 0.12)), x, y, a)
        ax = x0 + 150
        trans = [
            (nodes["SEARCHING"], nodes["LOCKED"], f"conf ≥ {fmt_fr(d['lock_conf'], 1)} pendant {d['min_frames']} images", 1.6),
            (nodes["LOCKED"], nodes["COASTING"], "cible perdue : dernière position gardée", 2.2),
        ]  # fmt: skip
        for (xa, ya), (xb, yb), lab, t0 in trans:
            p = ease_in_out(prog(t, t0, 0.5))
            arrow(fr, ax, ya + 92, ax, yb - 14, C["muted"], p, 1.0, 4, 14)
            blit(fr, para(lab, 40, 600, C["muted"], 450, lh=1.2), ax + 40, (ya + yb + 78) / 2, appear(t, t0 + 0.2, 0.5),
                 "lm")  # fmt: skip
        a3 = appear(t, 2.9, 0.5)
        yc = nodes["COASTING"][1] + 120
        blit(fr, para(f"retrouvée → LOCKED\nabsente plus de {d['coast']} images → SEARCHING", 40, 760,
                      C["muted"], 450, lh=1.25), x0 + 20, yc, a3)  # fmt: skip
        # Erreur de visée (colonne droite) sur une vraie frame.
        xi, yi = W - MX - self.iw, 300
        ai = appear(t, 3.6, 0.6)
        paste_rounded(fr, self.img, xi, yi + slide(t, 3.6), 16, ai)
        yi += slide(t, 3.6)
        k = self.iw / self.sw
        tg = self.f["tgt"]
        cx, cy = (tg[1] + tg[3]) / 2 * k + xi, (tg[2] + tg[4]) / 2 * k + yi
        mx, my = xi + self.iw / 2, yi + self.ih / 2
        p = ease_in_out(prog(t, 4.4, 0.8))
        blit(fr, reticle_img(34), mx, my, ai, "mm")
        bw = max(28, (tg[3] - tg[1]) * k * 1.6)
        blit(fr, brackets_img(bw, bw * 0.8, C["red"], 3), cx, cy, ai, "mm")
        arrow(fr, mx, my, mx + (cx - mx), my, C["dx"], p, ai, 4, 14, dash=8)
        arrow(fr, mx + (cx - mx), my, cx, cy, C["dy"], ease_in_out(prog(t, 5.0, 0.6)), ai, 4, 14, dash=8)
        ad = appear(t, 5.2, 0.5)
        blit(fr, text("dx", 40, C["dx"], 700, mono=True), (mx + cx) / 2, my + 12, ad, "mt")
        blit(fr, text("dy", 40, C["dy"], 700, mono=True), cx + 18, (my + cy) / 2, ad, "lm")
        af = appear(t, 5.8, 0.6)
        yf = yi + self.ih + 40
        blit(fr, text("dx = (cx − W/2) / (W/2)", 40, C["text"], 500, mono=True), xi, yf, af)
        blit(fr, text("dy = (cy − H/2) / (H/2)", 40, C["text"], 500, mono=True), xi, yf + 56, af)
        blit(fr, para("dans [−1, 1], écrits dans le CSV à chaque image", 40, self.iw, C["muted"], 450), xi, yf + 124,
             appear(t, 6.3, 0.6))  # fmt: skip
        return fr


class Data(Scene):
    """Données : vrais échantillons, sources et licences, découpage par séquence."""

    name, dur = "data", 10.0

    def setup(self):
        import csv as _csv

        rng = random.Random(7)
        rows = []
        for src in ("halmstad", "dut_track", "dut_det"):
            with open(ROOT / "data" / "yolo" / "manifests" / f"{src}.csv") as fh:
                rows += list(_csv.DictReader(fh))
        train = {p.name for p in (ROOT / "data" / "yolo" / "train" / "images").iterdir()}
        pick = []
        want = [("halmstad", "drone", 3), ("dut_track", None, 2), ("dut_det", None, 1), ("halmstad", "bird", 1),
                ("halmstad", "airplane", 1), ("halmstad", "helicopter", 1)]  # fmt: skip
        for src, cat, k in want:
            pool = [r for r in rows if r["source"] == src and (cat is None or r["category"] == cat)
                    and Path(r["image"]).name in train and (cat in ("bird", "airplane", "helicopter")
                    or int(r["n_boxes"]) > 0)]  # fmt: skip
            seqs = sorted({r["sequence"] for r in pool})
            rng.shuffle(seqs)
            for s in seqs[:k]:
                cands = [r for r in pool if r["sequence"] == s]
                pick.append(cands[len(cands) // 2])
        self.tiles = []
        tw, th = 316, 212
        self.tw, self.th = tw, th
        for r in pick:
            name = Path(r["image"]).name
            ip = ROOT / "data" / "yolo" / "train" / "images" / name
            img = load_image(ip)
            lbl = ip.parent.parent / "labels" / (ip.stem + ".txt")
            boxes = []
            if lbl.exists():
                for line in lbl.read_text().split("\n"):
                    if line.strip():
                        _, cx, cy, bw, bh = map(float, line.split())
                        boxes.append(((cx - bw / 2) * img.width, (cy - bh / 2) * img.height,
                                      (cx + bw / 2) * img.width, (cy + bh / 2) * img.height))  # fmt: skip
            if boxes:
                th_img, tf, _ = thumb_with_target(img, boxes[0], tw, th, zoom_box=4.0, min_crop=120)
                arr = np.asarray(th_img).copy()
                for b in boxes:
                    (a, bb), (c, e) = tf(b[0], b[1]), tf(b[2], b[3])
                    cv2.rectangle(arr, (int(a), int(bb)), (int(c), int(e)), C["green"], 2, cv2.LINE_AA)
                th_img = Image.fromarray(arr)
            else:
                th_img = cover(img, tw, th)
            self.tiles.append(th_img)

    def draw(self, t):
        fr = new_frame()
        header(fr, t, "LES DONNÉES", f"{ff('n_total', 0)} images réelles, découpées par séquence")
        gx, gy, gap = MX, 262, 14
        for k, img in enumerate(self.tiles):
            r, c = divmod(k, 3)
            t0 = 0.5 + 0.08 * k
            paste_rounded(fr, img, gx + c * (self.tw + gap), gy + r * (self.th + gap) + slide(t, t0, 20), 12,
                          appear(t, t0, 0.5))  # fmt: skip
        blit(fr, text("échantillons d'entraînement · vérité terrain en vert", 40, C["muted"], 450), gx,
             gy + 3 * (self.th + gap) + 8, appear(t, 1.6, 0.5))  # fmt: skip
        x = gx + 3 * self.tw + 2 * gap + 72
        cw = W - MX - x
        srcs = [("Halmstad (Svanström et al.)", "CC0-1.0 · 640×512 · avec oiseaux, avions et hélicoptères"),
                ("DUT Anti-UAV (Zhao et al.)", "Apache-2.0 · 720p et 1080p")]  # fmt: skip
        y = 268
        for k, (name, sub) in enumerate(srcs):
            t0 = 1.4 + 0.4 * k
            a = appear(t, t0, 0.5)
            sp_n, sp_s = text(name, 44, C["text"], 650), para(sub, 40, cw, C["muted"], 450, lh=1.2)
            blit(fr, sp_n, x, y + slide(t, t0), a)
            blit(fr, sp_s, x, y + sp_n.h + 8 + slide(t, t0), a)
            y += sp_n.h + 8 + sp_s.h + 36
        # Barre de découpage proportionnelle + légende.
        yb = y + 10
        a = appear(t, 2.8, 0.5)
        parts = [("train", "n_train", C["muted"]), ("val", "n_val", C["dx"]), ("test", "n_test", C["red"])]
        tot = sum(F.v(k) for _, k, _ in parts)
        p = ease_in_out(prog(t, 2.8, 1.0))
        xx = x
        for lab, key, col in parts:
            ww = (cw - 2 * 6) * F.v(key) / tot
            blit(fr, rrect(max(4, int(ww * p)), 30, 6, fill=col), xx, yb, a)
            xx += ww + 6
        ys = yb + 58
        for k, (lab, key, col) in enumerate(parts):
            ak = appear(t, 3.2 + 0.2 * k, 0.4)
            blit(fr, rrect(22, 22, 5, fill=col), x, ys + k * 54 + 12, ak)
            blit(fr, text(lab, 40, C["text"], 500, mono=True), x + 40, ys + k * 54, ak)
            blit(fr, text(ff(key, 0), 40, C["text"], 600, mono=True), x + 330, ys + k * 54, ak, "rt")
        a2 = appear(t, 4.4, 0.6)
        blit(fr, para("Découpage par séquence : aucune vidéo du test n'a servi à l'entraînement.", 44, cw,
                      C["text"], 500), x, ys + 3 * 54 + 34 + slide(t, 4.4), a2)  # fmt: skip
        return fr


class Demo(Scene):
    """Séquence de test : vidéo avec overlay, loupe sur la cible, graphe dx/dy synchronisé."""

    def __init__(self, name, stem, title, dur, start=None, lead=1.0, lo=0):
        self.name, self.stem, self.title, self.dur = name, stem, title, dur
        self.start, self.lead, self.lo = start, lead, lo

    def setup(self):
        self.clip = c = TrackClip(self.stem)
        self.n = int(self.dur * FPS)
        self.a = self.start if self.start is not None else c.pick_segment(self.n, int(self.lead * FPS), self.lo)
        ph = 652
        pw = int(round(ph * c.w / c.h))
        if pw > 1160:
            pw = 1160
            ph = int(round(pw * c.h / c.w))
        self.pw, self.ph = pw, ph
        self.px, self.py = MX, 236
        self.rx = self.px + pw + 40
        self.rw = W - MX - self.rx
        self.loupe_h = int(min(self.rw * 0.62, ph * 0.5))
        self.graph = AimGraph(c, self.a, self.n, self.rw, ph - self.loupe_h - 24)
        self.cent = c.centers(self.a, self.a + self.n)
        self.rows = [c.f(self.a + j) for j in range(self.n)]
        sizes = sorted(s for _, _, s in self.cent)
        med = sizes[len(sizes) // 2]
        # Recadrage de la loupe : la cible occupe environ 1/3 de la hauteur (px source).
        self.crop_w = max(60.0, med * 3.0) * self.rw / self.loupe_h

    def loupe(self, i, j) -> Image.Image:
        c = self.clip
        lw, lh = self.rw, self.loupe_h
        cx, cy, _ = self.cent[j]
        cw = self.crop_w
        ch = cw * lh / lw
        ox, oy = clamp(cx - cw / 2, 0, c.w - cw), clamp(cy - ch / 2, 0, c.h - ch)
        raw = c.raw(i)
        crop = raw[int(oy) : int(oy + ch), int(ox) : int(ox + cw)]
        img = cv2.resize(crop, (lw, lh), interpolation=cv2.INTER_CUBIC)
        pil = bgr_to_pil(img)
        f = c.f(i)
        s = lw / cw
        if f["tgt"]:
            x1, y1, x2, y2 = ((f["tgt"][1] - ox) * s, (f["tgt"][2] - oy) * s, (f["tgt"][3] - ox) * s,
                              (f["tgt"][4] - oy) * s)  # fmt: skip
            pad = 0.25 * max(x2 - x1, y2 - y1)
            col = STATE_RGB[f["state"]]
            blit(pil, brackets_img(x2 - x1 + 2 * pad, y2 - y1 + 2 * pad, col, 4), (x1 + x2) / 2, (y1 + y2) / 2, 1, "mm")
        blit(pil, pill(f"LOUPE ×{s:.0f}", 40, C["text"], fill=C["bg"], outline=False), 16, 16, 0.85)
        st = f["state"]
        blit(pil, pill(st, 40, STATE_RGB[st], fill=C["bg"]), 20, lh - 20, 1, "lb")
        return pil

    def draw(self, t):
        fr = new_frame()
        j = min(self.n - 1, int(t * FPS))
        i = self.a + j
        header(fr, t, "DÉMO · SÉQUENCE DE TEST JAMAIS VUE", self.title, t0=0.0, size=60)
        av = appear(t, 0.15, 0.5)
        vid = self.clip.overlay(i, self.pw, self.ph)
        blit(fr, shadow(self.pw, self.ph), self.px, self.py, av)
        paste_rounded(fr, vid, self.px, self.py, 16, av)
        ar = appear(t, 0.45, 0.5)
        lp = self.loupe(i, j)
        paste_rounded(fr, lp, self.rx, self.py, 16, ar)
        blit(fr, rrect(self.rw, self.loupe_h, 16, outline=C["line"], width=2), self.rx, self.py, ar)
        self.graph.draw(fr, self.rx, self.py + self.loupe_h + 24, j, appear(t, 0.65, 0.5))
        state_legend(fr, self.rows, j, self.px, self.py + self.ph + 30, appear(t, 0.8, 0.5))
        return fr


class Birds(Scene):
    """Oiseaux : R1 (640) contre R2 (1280) côte à côte, même séquence de test, puis bilan chiffré."""

    name, dur = "birds", 10.0
    stem, tags = "halmstad__V_BIRD_020", ("r1", "r2")

    def setup(self):
        self.r1 = TrackClip(self.stem, self.tags[0])
        self.r2 = TrackClip(self.stem, self.tags[1])
        # Fenêtre vidéo : début de la séquence, au plus 6,5 s, et arrêtée avant un éventuel verrou de R2
        # (avec le .pt sur CPU, R2 se verrouillait sur des branches fixes vers la frame 190 ; avec
        # l'engine des clips de démo, jamais). La note de fin de scène est choisie selon les pistes.
        self.first_r2_lock = next((f["i"] for f in self.r2.frames if f["state"] != "SEARCHING"), len(self.r2))
        self.n = max(FPS, min(int(6.5 * FPS), self.first_r2_lock - 4))
        self.a = 0
        self.t_vid = self.n / FPS
        self.ph = 572
        self.pw = int(round(self.ph * self.r1.w / self.r1.h))
        locks = [sum(c.f(q)["state"] == "LOCKED" for q in range(self.n)) for c in (self.r1, self.r2)]
        ids_r1 = {self.r1.f(q)["tgt"][0] for q in range(self.n) if self.r1.f(q)["state"] == "LOCKED"}
        first_r1 = next((q for q in range(self.n) if self.r1.f(q)["state"] == "LOCKED"), 0)
        self.t_cap = max(1.2, first_r1 / FPS + 0.3)  # la phrase arrive avec le premier verrou de R1
        n_all = len(self.r2)
        r2_all = sum(f["state"] != "SEARCHING" for f in self.r2.frames)
        if r2_all == 0:
            self.note = f"Sur toute la séquence ({n_all} images), R2 ne se verrouille jamais."
        else:
            self.note = "Plus loin dans cette séquence, R2 se verrouille sur un objet fixe : voir les limites."
        if locks[0] > 0 and locks[1] == 0:
            cible = "des oiseaux" if len(ids_r1) > 1 else "un oiseau"
            self.cap = f"Même séquence, même suivi : R1 se verrouille sur {cible}, R2 ne se verrouille pas."
        elif locks[0] == 0 and locks[1] == 0:
            self.cap = "Même séquence, même suivi : aucun des deux modèles ne se verrouille sur un oiseau."
        else:
            self.cap = "Même séquence de test, même suivi, deux modèles."

    def draw(self, t):
        fr = new_frame()
        j = min(self.n - 1, int(t * FPS))
        i = self.a + j
        header(fr, t, "DÉMO · OISEAUX (TEST)", "Un oiseau n'est pas un drone", size=60)
        gap = 96
        x0 = (W - 2 * self.pw - gap) / 2
        dim = 1 - 0.75 * ease_in_out(prog(t, self.t_vid, 0.5))  # vidéos atténuées pendant le bilan
        for k, (clip, lab) in enumerate(((self.r1, "R1 · entraîné à 640\u00a0px"),
                                         (self.r2, "R2 · affiné à 1280\u00a0px"))):  # fmt: skip
            x = x0 + k * (self.pw + gap)
            y = 296
            a = appear(t, 0.2 + 0.25 * k, 0.5)
            blit(fr, text(lab, 44, C["text"], 650), x, 230, a)
            blit(fr, shadow(self.pw, self.ph), x, y, a * dim)
            paste_rounded(fr, clip.overlay(i, self.pw, self.ph), x, y, 16, a * dim)
            n_lock = sum(clip.f(q)["state"] == "LOCKED" for q in range(self.a, i + 1))
            col = C["red"] if n_lock else C["green"]
            lab_n = f"verrouillé : {n_lock} / {j + 1} images"
            blit(fr, text(lab_n, 40, col, 600, mono=True), x, y + self.ph + 16, a)
        blit(
            fr,
            text(self.cap, 44, C["text"], 500),
            W / 2,
            H - MY - 44,
            appear(t, self.t_cap, 0.4, self.t_vid + 0.2),
            "mt",
        )
        # Bilan sur tout le test (fichiers de résultats), par-dessus les vidéos atténuées.
        t2 = self.t_vid + 0.2
        a2 = appear(t, t2, 0.5)
        if a2 > 0:
            pw_, ph_ = 1240, 360
            px, py = (W - pw_) / 2, 300 + slide(t, t2, 20)
            panel(fr, px, py, pw_, ph_, a2, fill=C["panel"])
            n_img = F.v("n_bird_img")
            blit(fr, text(f"Fausses détections sur les {n_img} images d'oiseaux du test", 44, C["text"], 650),
                 px + 48, py + 36, a2)  # fmt: skip
            bx, bw = px + 220, pw_ - 220 - 160
            for k, (lab, key, col) in enumerate((("R1", "r1_fp_bird", C["muted"]), ("R2", "r2_fp_bird", C["red"]))):
                yy = py + 140 + k * 92
                p = ease_out(prog(t, t2 + 0.3 + 0.25 * k, 0.8))
                val = F.v(key)
                blit(fr, text(lab, 44, C["text"], 700, mono=True), px + 48, yy + 30, a2, "lm")
                blit(fr, rrect(int(bw), 56, 8, fill=C["panel2"]), bx, yy + 2, a2)
                blit(fr, rrect(max(8, int(bw * min(1.0, val / n_img) * p)), 56, 8, fill=col), bx, yy + 2, a2)
                blit(fr, text(f"{round(val * p)}", 48, C["text"], 700, mono=True), bx + bw + 28, yy + 30, a2, "lm")
            note = self.note
            blit(fr, text(note, 40, C["muted"], 450), W / 2, H - MY - 44, appear(t, t2 + 1.2, 0.5), "mt")
        return fr


class Results(Scene):
    """Chiffres clés du test + SAHI sur 1080p."""

    name, dur = "results", 14.0

    def draw(self, t):
        fr = new_frame()
        header(fr, t, "RÉSULTATS · TEST", f"{ff('n_test_img', 0)} images, {ff('n_test_seq', 0)} séquences jamais vues")
        tiles = [
            (F.v("r30"), 2, "rappel à IoU\u00a00,3", "la cible est trouvée"),
            (F.v("p30"), 2, "précision à IoU\u00a00,3", "peu de fausses alertes"),
            (F.v("map50"), 2, "mAP50", f"mAP50-95 : {ff('map5095', 3)}"),
        ]
        fps = F.v("trt_fps_video")
        if fps:
            tiles.append((fps, 0, "images/s de bout en bout", "TensorRT FP16, vidéo 1080p, suivi compris"))
        else:
            tiles.append((F.v("lat_ms_val"), 0, "ms par image", "PyTorch, RTX 3070 Laptop"))
        n = len(tiles)
        tw = 408
        gap = (W - 2 * MX - n * tw) / (n - 1)
        y = 262
        end_a = 7.6
        blocks = [(para(lab, 44, tw - 56, C["text"], 600, lh=1.18), para(sub, 40, tw - 56, C["muted"], 450, lh=1.2))
                  for _, _, lab, sub in tiles]  # fmt: skip
        th = 24 + 150 + max(b1.h + 12 + b2.h for b1, b2 in blocks) + 34
        for k, ((val, nd, _, _), (sp_l, sp_s)) in enumerate(zip(tiles, blocks)):
            t0 = 0.6 + 0.35 * k
            a = appear(t, t0, 0.5, end_a + 0.6)
            x = MX + k * (tw + gap)
            dy = slide(t, t0, 26)
            panel(fr, x, y + dy, tw, th, a)
            blit(fr, rrect(tw, 4, 0, fill=C["red"]), x, y + dy, a * 0.9)
            p = ease_out(prog(t, t0 + 0.1, 1.1))
            blit(fr, text(fmt_fr(val * p, nd), 120, C["text"], 700, mono=True), x + 26, y + 18 + dy, a)
            blit(fr, sp_l, x + 28, y + 174 + dy, a)
            blit(fr, sp_s, x + 28, y + 174 + sp_l.h + 12 + dy, a)
        note = ("IoU 0,3 : sur des cibles de 15 à 40\u00a0px la boîte est approximative, mais la visée n'utilise "
                "que son centre.")  # fmt: skip
        sp_note = para(note, 44, W - 2 * MX, C["muted"], 450)
        blit(fr, sp_note, MX, y + th + 44 + slide(t, 2.6), appear(t, 2.6, 0.6, end_a + 0.6))
        if fps:
            hw = "Mesures : RTX 3070 Laptop (PC portable), pas encore sur Jetson."
            blit(fr, text(hw, 40, C["dim"], 450), MX, y + th + 44 + sp_note.h + 30, appear(t, 3.2, 0.6, end_a + 0.6))
        # Partie 2 : SAHI sur les images 1080p (DUT).
        t2 = end_a + 0.7
        a = appear(t, t2, 0.5)
        if a > 0:
            yb = 300
            blit(fr, text("Images 1080p (DUT) : découper en tuiles (SAHI) aide les petites cibles", 48, C["text"], 650),
                 MX, yb + slide(t, t2), a)  # fmt: skip
            bars = [("image entière", F.v("dut_map50"), F.v("dut_full_ms"), C["muted"]),
                    ("tuiles SAHI 640", F.v("dut_sahi_map50"), F.v("dut_sahi_ms"), C["red"])]  # fmt: skip
            bx, bw_max = MX + 420, 900
            for k, (lab, m, ms, col) in enumerate(bars):
                yy = yb + 130 + k * 120
                ak = appear(t, t2 + 0.3 + 0.3 * k, 0.5)
                p = ease_out(prog(t, t2 + 0.4 + 0.3 * k, 0.9))
                blit(fr, text(lab, 44, C["text"], 550), MX, yy + 40, ak, "lm")
                blit(fr, rrect(max(6, int(bw_max * m * p)), 64, 8, fill=col), bx, yy + 8, ak)
                blit(
                    fr,
                    text(fmt_fr(m * p, 3), 48, C["text"], 700, mono=True),
                    bx + bw_max * m * p + 24,
                    yy + 40,
                    ak,
                    "lm",
                )
                blit(fr, text(f"{fmt_fr(ms, 0)} ms/image", 40, C["muted"], 500, mono=True), W - MX, yy + 40, ak, "rm")
            blit(fr, text("mAP50, test DUT 1080p", 40, C["dim"], 450), bx, yb + 380, appear(t, t2 + 1.0, 0.5))
            blit(fr, para("×3 en temps de calcul : à réserver à une zone d'intérêt autour de la cible.", 44,
                          W - 2 * MX, C["muted"], 450), MX, yb + 470, appear(t, t2 + 1.6, 0.6))  # fmt: skip
        return fr


class Limits(Scene):
    """Limites honnêtes (avec les deux faux verrous fixes, en images) puis prochaines étapes."""

    name, dur = "limits", 14.0
    t_next = 6.8  # bascule de la colonne de droite : faux verrous → prochaines étapes

    def setup(self):
        self.lim = [
            f"Boîtes approximatives sur les cibles de moins de 20\u00a0px (mAP50-95 : {ff('map5095', 3)})",
            "Faux positifs sur des objets fixes",
            "Contre-jour, nuit et infrarouge : non testés",
            "Pas encore de vidéo FPV réelle",
        ]
        self.next = [
            "Jetson Orin + TensorRT (export ONNX prêt)",
            "Filtre de mouvement contre les faux positifs fixes",
            "Fusion audio + RF pour la détection lointaine",
            "Guidage terminal : filtrage, prédiction, compensation de latence",
        ]
        tw, th = 392, 260
        self.tw, self.th = tw, th
        # Faux positifs fixes de R2 (galerie d'échecs de l'évaluation, zoom de droite sans le bandeau de titre).
        imgs = []
        for name, cy in (("fp_bird__halmstad__V_BIRD_020__000180__90_433.jpg", 1.0),
                         ("fp_sky_small__dut_track__video15__000960__1345_487.jpg", 0.45)):  # fmt: skip
            g = load_image(ROOT / "docs/assets/failures" / name)
            imgs.append(cover(g.crop((g.width - 300, 22, g.width, g.height)), tw, th, 0.5, cy))
        im1, im2 = imgs
        self.fp = [
            (im1, "Branches en fleurs (Halmstad, séquence d'oiseaux)"),
            (im2, "Élément fixe sur un toit (DUT video15)"),
        ]

    def draw(self, t):
        fr = new_frame()
        header(fr, t, "LIMITES ET PISTES", "Ce qui marche, ce qui manque")
        colw = (W - 2 * MX - 80) // 2
        xr = MX + colw + 80
        self._column(fr, t, MX, colw, "Limites", self.lim, C["amber"], 0.6, None)
        # Colonne de droite, phase 1 : les deux faux verrous, en vraies images.
        a1 = appear(t, 1.6, 0.5, self.t_next)
        if a1 > 0:
            blit(fr, text("Faux positifs fixes, en images", 48, C["amber"], 700), xr, 270 + slide(t, 1.6), a1)
            blit(fr, rrect(colw, 4, 0, fill=C["amber"]), xr, 340, a1)
            for m, (img, cap) in enumerate(self.fp):
                x = xr + m * (self.tw + (colw - 2 * self.tw))
                am = appear(t, 2.0 + 0.4 * m, 0.5, self.t_next)
                paste_rounded(fr, img, x, 380 + slide(t, 2.0 + 0.4 * m, 16), 12, am)
                blit(fr, para(cap, 40, self.tw, C["muted"], 450, lh=1.2), x, 380 + self.th + 18, am)
        # Phase 2 : prochaines étapes.
        self._column(fr, t, xr, colw, "Prochaines étapes", self.next, C["dx"], self.t_next + 0.2, None)
        return fr

    @staticmethod
    def _column(fr, t, x, colw, title, items, col, t0, end):
        a = appear(t, t0, 0.5, end)
        if a <= 0 and t < t0:
            return
        blit(fr, text(title, 48, col, 700), x, 270 + slide(t, t0), a)
        blit(fr, rrect(colw, 4, 0, fill=col), x, 340, a)
        y = 380
        for m, s in enumerate(items):
            tm = t0 + 0.4 + 0.55 * m
            am = appear(t, tm, 0.5, end)
            sp = para(s, 44, colw - 50, C["text"], 450)
            blit(fr, circle(7, fill=col), x + 8, y + 30, am, "mm")
            blit(fr, sp, x + 40, y + slide(t, tm, 14), am)
            y += sp.h + 40


class End(Scene):
    """Fin : nom du dépôt, auteur, crédits."""

    name, dur, fade_out = "end", 7.0, 0.6

    def draw(self, t):
        fr = new_frame()
        title = text("drone-detect", 120, C["text"], 760)
        cx, cy = W / 2, 400
        a = appear(t, 0.3, 0.6)
        blit(fr, title, cx, cy, a, "mm")
        # Coins de verrouillage qui se referment sur le titre.
        p = ease_out(prog(t, 0.6, 0.9))
        bw, bh = title.w + 120 + (1 - p) * 300, title.h + 60 + (1 - p) * 140
        col = mix(C["yellow"], C["red"], prog(t, 1.3, 0.25))
        blit(fr, brackets_img(bw, bh, col, 5, 0.22), cx, cy, appear(t, 0.5, 0.4), "mm")
        al = appear(t, 1.5, 0.4)
        lk = pill("LOCKED", 40, C["red"], fill=mix(C["bg"], C["red"], 0.12))
        blit(fr, lk, cx + bw / 2, cy - bh / 2 - 28, al, "rb")
        blit(
            fr,
            text("github.com/<pseudo>/drone-detect", 48, C["text"], 500, mono=True),
            cx,
            cy + 150,
            appear(t, 1.8, 0.5),
            "mm",
        )
        blit(fr, text("Nathan · IA / data · pilote et constructeur FPV", 48, C["text"], 500), cx, cy + 240,
             appear(t, 2.2, 0.5), "mm")  # fmt: skip
        cred = ("Données : Halmstad (CC0-1.0), DUT Anti-UAV (Apache-2.0)\n"
                "Ultralytics YOLO11, ByteTrack · code sous licence AGPL-3.0")  # fmt: skip
        blit(fr, para(cred, 40, W - 2 * MX, C["dim"], 450, align="center"), cx, H - MY - 40, appear(t, 3.0, 0.6), "mb")
        return fr


def all_scenes() -> list[Scene]:
    return [
        Hook(),
        Context(),
        Pipeline(),
        Lock(),
        Data(),
        # Fenêtre 632–962 : verrou #1 continu (vérifié à l'œil), dx de +0,21 à −0,55, descente vers le toit.
        Demo("demo19", "dut_track__video19", "DUT Anti-UAV video19 · 1080p · au-dessus d'un toit", 11.0, start=632),
        Demo("demo35", "halmstad__V_DRONE_035", "Halmstad V_DRONE_035 · 640×512 · ciel couvert", 10.0),
        Birds(),
        Results(),
        Limits(),
        End(),
    ]  # fmt: skip
