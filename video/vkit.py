"""Boîte à outils de rendu de la vidéo de présentation (Pillow + OpenCV, sans dépendance en plus).

Une frame est une ``PIL.Image`` RGB 1920×1080. Les éléments (textes, formes) sont des
*sprites* RGBA mis en cache, posés avec une opacité (``blit``). Les formes sont dessinées
en sur-échantillonnage ×4 puis réduites, pour un anticrénelage propre ; les tracés de
courbes passent par OpenCV (``LINE_AA``, coordonnées sous-pixel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "video" / "fonts"
W, H, FPS = 1920, 1080, 30
MX, MY = 96, 72  # marges de sécurité (5 % / 6,7 %)

# Palette (RGB). Couleurs d'état = celles de l'overlay de scripts/demo.py (BGR → RGB).
C = {
    "bg": (9, 12, 17),
    "bg2": (14, 18, 25),
    "panel": (20, 25, 33),
    "panel2": (27, 33, 43),
    "line": (48, 57, 70),
    "grid": (33, 40, 50),
    "text": (236, 240, 245),
    "muted": (150, 160, 174),
    "dim": (98, 108, 122),
    "red": (255, 40, 40),  # LOCKED
    "amber": (255, 150, 0),  # COASTING
    "yellow": (255, 220, 0),  # SEARCHING
    "cyan": (0, 200, 255),  # pistes non verrouillées
    "dx": (57, 135, 229),  # série dx (validée daltonisme avec dy sur fond sombre)
    "dy": (25, 158, 112),  # série dy
    "green": (70, 200, 120),
}
STATE_RGB = {"SEARCHING": C["yellow"], "LOCKED": C["red"], "COASTING": C["amber"]}


# --- Temps et interpolations ----------------------------------------------------------


def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return a if x < a else b if x > b else x


def prog(t: float, start: float, dur: float) -> float:
    """Avancement linéaire 0→1 entre ``start`` et ``start + dur``."""
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


def ease_out(p: float) -> float:
    return 1 - (1 - clamp(p)) ** 3


def ease_in_out(p: float) -> float:
    p = clamp(p)
    return 4 * p**3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def appear(t: float, start: float, dur: float = 0.5, end: float | None = None, out: float = 0.35) -> float:
    """Opacité d'un élément qui apparaît à ``start`` (fondu ``dur``) et disparaît à ``end``."""
    a = ease_out(prog(t, start, dur))
    if end is not None:
        a *= 1 - ease_in_out(prog(t, end - out, out))
    return a


def lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p


def mix(c1, c2, p: float) -> tuple:
    return tuple(int(round(lerp(a, b, p))) for a, b in zip(c1, c2))


# --- Polices et texte -----------------------------------------------------------------


@lru_cache(maxsize=256)
def font(size: int, weight: int = 400, mono: bool = False) -> ImageFont.FreeTypeFont:
    name = "JetBrainsMono-Variable.ttf" if mono else "Inter-Variable.ttf"
    path = FONT_DIR / name
    if not path.exists():
        raise SystemExit(f"police absente : {path} — lancer .venv/bin/python video/fetch_fonts.py")
    f = ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_axes([weight] if mono else [clamp(size, 14, 32), weight])
    return f


@dataclass(frozen=True)
class Sprite:
    """Image RGBA + origine logique (``ox``, ``oy``) et taille logique (``w``, ``h``)."""

    img: Image.Image
    ox: int = 0
    oy: int = 0
    w: int = 0
    h: int = 0


@lru_cache(maxsize=8192)
def text(s: str, size: int, color=C["text"], weight: int = 400, mono: bool = False, track: float = 0.0) -> Sprite:
    """Une ligne de texte. Position logique : coin haut-gauche de la ligne (ascendante)."""
    f = font(size, weight, mono)
    asc, desc = f.getmetrics()
    pad = int(size * 0.3)
    if track:
        widths = [f.getlength(ch) + track * size for ch in s]
        lw = int(math.ceil(sum(widths) - (track * size if s else 0)))
    else:
        lw = int(math.ceil(f.getlength(s)))
    img = Image.new("RGBA", (lw + 2 * pad, asc + desc + 2 * pad), (*color, 0))
    d = ImageDraw.Draw(img)
    if track:
        x = pad
        for ch, cw in zip(s, widths):
            d.text((x, pad), ch, font=f, fill=(*color, 255), anchor="la")
            x += cw
    else:
        d.text((pad, pad), s, font=f, fill=(*color, 255), anchor="la")
    return Sprite(img, pad, pad, lw, asc + desc)


def wrap(s: str, size: int, max_w: int, weight: int = 400, mono: bool = False) -> list[str]:
    """Coupe ``s`` en lignes de largeur ≤ ``max_w`` (les ``\\n`` forcent un retour)."""
    f = font(size, weight, mono)
    lines = []
    for para in s.split("\n"):
        cur = ""
        for word in para.split(" "):
            cand = f"{cur} {word}" if cur else word
            if f.getlength(cand) <= max_w or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


@lru_cache(maxsize=1024)
def para(s: str, size: int, max_w: int, color=C["text"], weight: int = 400, lh: float = 1.28,
         align: str = "left", mono: bool = False) -> Sprite:  # fmt: skip
    """Paragraphe multi-lignes (``align`` : left | center)."""
    lines = wrap(s, size, max_w, weight, mono)
    sprites = [text(line, size, color, weight, mono) for line in lines]
    step = int(round(size * lh))
    lw = max(sp.w for sp in sprites)
    pad = sprites[0].ox
    hh = step * (len(lines) - 1) + sprites[-1].h
    img = Image.new("RGBA", (lw + 2 * pad, hh + 2 * pad), (*color, 0))
    for i, sp in enumerate(sprites):
        x = 0 if align == "left" else (lw - sp.w) // 2
        img.alpha_composite(sp.img, (x, i * step))
    return Sprite(img, pad, pad, lw, hh)


# --- Formes ---------------------------------------------------------------------------

SS = 4  # sur-échantillonnage des formes


def _downsample(big: Image.Image, w: int, h: int) -> Image.Image:
    return big.convert("RGBa").resize((w, h), Image.Resampling.LANCZOS).convert("RGBA")


@lru_cache(maxsize=512)
def rrect(w: int, h: int, r: int, fill=None, outline=None, width: int = 2, dash: int = 0) -> Sprite:
    """Rectangle arrondi (remplissage et/ou contour, contour pointillé si ``dash`` > 0)."""
    w, h = max(2, int(w)), max(2, int(h))
    big = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    box = [0, 0, w * SS - 1, h * SS - 1]
    if fill is not None:
        d.rounded_rectangle(box, r * SS, fill=_rgba(fill))
    if outline is not None and not dash:
        inset = width * SS / 2
        d.rounded_rectangle([inset, inset, w * SS - 1 - inset, h * SS - 1 - inset], max(0, r * SS - inset),
                            outline=_rgba(outline), width=width * SS)  # fmt: skip
    elif outline is not None:
        _dashed_rrect(d, w * SS, h * SS, r * SS, width * SS, dash * SS, _rgba(outline))
    return Sprite(_downsample(big, w, h), 0, 0, w, h)


def _rgba(c):
    return tuple(c) if len(c) == 4 else (*c, 255)


def _dashed_rrect(d, w, h, r, width, dash, color):
    """Contour pointillé approximé : segments sur le périmètre du rectangle arrondi."""
    pts = []
    inset = width / 2
    x0, y0, x1, y1 = inset, inset, w - inset, h - inset
    rr = max(1, r - inset)
    n_arc = 12
    corners = [(x1 - rr, y0 + rr, -90), (x1 - rr, y1 - rr, 0), (x0 + rr, y1 - rr, 90), (x0 + rr, y0 + rr, 180)]
    for cx, cy, a0 in corners:
        for k in range(n_arc + 1):
            a = math.radians(a0 + 90 * k / n_arc)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    pts.append(pts[0])
    # Parcours à abscisse curviligne, alternance trait / vide.
    seg_len, on, acc = dash, True, 0.0
    cur = [pts[0]]
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        L = math.hypot(bx - ax, by - ay)
        pos = 0.0
        while pos < L:
            step = min(seg_len - acc, L - pos)
            pos += step
            acc += step
            px, py = ax + (bx - ax) * pos / L, ay + (by - ay) * pos / L
            if on:
                cur.append((px, py))
            if acc >= seg_len - 1e-6:
                if on and len(cur) > 1:
                    d.line(cur, fill=color, width=int(width), joint="curve")
                on, acc, cur = not on, 0.0, [(px, py)]
    if on and len(cur) > 1:
        d.line(cur, fill=color, width=int(width), joint="curve")


@lru_cache(maxsize=256)
def circle(r: int, fill=None, outline=None, width: int = 2) -> Sprite:
    s = 2 * r + 2
    big = Image.new("RGBA", (s * SS, s * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    box = [SS, SS, (s - 1) * SS, (s - 1) * SS]
    d.ellipse(box, fill=_rgba(fill) if fill else None, outline=_rgba(outline) if outline else None,
              width=width * SS)  # fmt: skip
    return Sprite(_downsample(big, s, s), r + 1, r + 1, s, s)


def vector(w: int, h: int, draw_fn, scale: int = SS) -> Image.Image:
    """Calque RGBA ``w``×``h`` dessiné par ``draw_fn(ImageDraw, k)`` en coordonnées ×k (k = SS)."""
    big = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), scale)
    return _downsample(big, w, h)


@lru_cache(maxsize=64)
def gradient_v(w: int, h: int, top, bottom) -> Image.Image:
    """Dégradé vertical RGBA (couleurs avec alpha)."""
    t = np.linspace(0, 1, h)[:, None, None]
    a, b = np.array(_rgba(top), float), np.array(_rgba(bottom), float)
    arr = (a * (1 - t) + b * t).repeat(w, axis=1)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


@lru_cache(maxsize=4)
def background() -> Image.Image:
    """Fond sombre commun : léger dégradé radial + grille très discrète."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W * 0.5) / W) ** 2 + ((yy - H * 0.42) / H) ** 2)
    v = np.clip(1 - r * 1.25, 0, 1)[..., None]
    base, glow = np.array(C["bg"], np.float32), np.array(C["bg2"], np.float32) + 4
    arr = base * (1 - v) + glow * v
    step = 48
    grid = ((xx % step) < 1) | ((yy % step) < 1)
    arr[grid] += 3.5
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def new_frame() -> Image.Image:
    return background().copy()


# --- Composition ----------------------------------------------------------------------


def blit(frame: Image.Image, sp: Sprite | Image.Image, x: float, y: float, alpha: float = 1.0,
         anchor: str = "lt") -> None:  # fmt: skip
    """Pose un sprite sur ``frame`` (RGB) à la position logique (x, y) avec une opacité.

    ``anchor`` : 2 lettres, horizontal (l, m, r) puis vertical (t, m, b), sur la boîte logique.
    """
    if alpha <= 0.003:
        return
    if isinstance(sp, Image.Image):
        sp = Sprite(sp, 0, 0, sp.width, sp.height)
    ax, ay = anchor
    x -= {"l": 0, "m": sp.w / 2, "r": sp.w}[ax]
    y -= {"t": 0, "m": sp.h / 2, "b": sp.h}[ay]
    img = sp.img
    a = img.getchannel("A")
    if alpha < 0.997:
        a = a.point(lambda v, k=alpha: int(v * k + 0.5))
    frame.paste(img.convert("RGB") if img.mode != "RGB" else img, (int(round(x - sp.ox)), int(round(y - sp.oy))), a)


def paste_rgb(frame: Image.Image, img: Image.Image, x: int, y: int, alpha: float = 1.0) -> None:
    """Pose une image RGB opaque (vidéo, vignette) avec une opacité globale."""
    if alpha <= 0.003:
        return
    if alpha >= 0.997:
        frame.paste(img, (int(x), int(y)))
    else:
        frame.paste(img, (int(x), int(y)), Image.new("L", img.size, int(255 * alpha)))


def fade_frame(frame: Image.Image, k: float) -> Image.Image:
    """Fondu de toute l'image vers le fond (k = 1 : intacte, 0 : fond seul)."""
    if k >= 0.999:
        return frame
    return Image.blend(background(), frame, clamp(k))


def shadow(w: int, h: int, r: int = 18, blur: int = 22, alpha: int = 150) -> Sprite:
    """Ombre portée douce pour un panneau ``w``×``h``."""
    return _shadow(int(w), int(h), r, blur, alpha)


@lru_cache(maxsize=64)
def _shadow(w, h, r, blur, alpha):
    pad = blur * 2
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([pad, pad + 8, pad + w, pad + h + 8], r, fill=(0, 0, 0, alpha))
    return Sprite(img.filter(ImageFilter.GaussianBlur(blur)), pad, pad, w, h)


def pill(label: str, size: int, color, fill=None, weight: int = 650, mono: bool = True, padx: float = 0.55,
         outline: bool = True) -> Sprite:  # fmt: skip
    """Étiquette arrondie (état du verrou, tags)."""
    return _pill(label, size, tuple(color), tuple(fill) if fill else None, weight, mono, padx, outline)


@lru_cache(maxsize=256)
def _pill(label, size, color, fill, weight, mono, padx, outline):
    t = text(label, size, color, weight, mono)
    px, py = int(size * padx), int(size * 0.28)
    w, h = t.w + 2 * px, t.h + 2 * py
    bg = rrect(w, h, h // 2, fill=fill, outline=color if outline else None, width=3)
    img = bg.img.copy()
    img.alpha_composite(t.img, (px - t.ox, py - t.oy))
    return Sprite(img, 0, 0, w, h)


# --- Vidéo ----------------------------------------------------------------------------


class VideoReader:
    """Lecture séquentielle d'une vidéo (``get(i)`` avance sans re-décoder si possible)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise SystemExit(f"vidéo illisible : {self.path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = 0
        self.last = None

    def get(self, i: int) -> np.ndarray:
        """Frame ``i`` en BGR (bornée à la dernière frame)."""
        i = max(0, min(i, self.n - 1))
        if self.last is not None and i == self.pos - 1:
            return self.last
        if i != self.pos:
            if 0 < i - self.pos < 90:
                for _ in range(i - self.pos):
                    self.cap.grab()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i
        ok, f = self.cap.read()
        if not ok:
            raise SystemExit(f"lecture impossible : {self.path} frame {i}")
        self.pos, self.last = i + 1, f
        return f


def bgr_to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def cover(img: Image.Image, w: int, h: int, cx: float = 0.5, cy: float = 0.5, zoom: float = 1.0) -> Image.Image:
    """Recadre et redimensionne ``img`` pour remplir ``w``×``h`` (centre relatif ``cx``, ``cy``)."""
    iw, ih = img.size
    s = max(w / iw, h / ih) * zoom
    cw, ch = w / s, h / s
    x0 = clamp(cx * iw - cw / 2, 0, iw - cw)
    y0 = clamp(cy * ih - ch / 2, 0, ih - ch)
    return img.resize((w, h), Image.Resampling.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))


def rounded_mask(w: int, h: int, r: int) -> Image.Image:
    return _rounded_mask(int(w), int(h), int(r))


@lru_cache(maxsize=64)
def _rounded_mask(w, h, r):
    return rrect(w, h, r, fill=(255, 255, 255)).img.getchannel("A")


def paste_rounded(frame: Image.Image, img: Image.Image, x: int, y: int, r: int = 16, alpha: float = 1.0) -> None:
    """Pose une image RGB aux coins arrondis."""
    m = rounded_mask(img.width, img.height, r)
    if alpha < 0.997:
        m = m.point(lambda v, k=alpha: int(v * k + 0.5))
    frame.paste(img, (int(x), int(y)), m)


def cv_polyline(arr: np.ndarray, pts, color, thick: int = 2) -> None:
    """Polyligne anticrénelée sous-pixel (``arr`` RGB numpy, ``pts`` liste de (x, y) flottants)."""
    if len(pts) < 2:
        return
    p = (np.asarray(pts, np.float64) * 16).round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(arr, [p], False, tuple(int(c) for c in color), thick, cv2.LINE_AA, shift=4)


def cv_circle(arr: np.ndarray, x: float, y: float, r: float, color, thick: int = -1) -> None:
    cv2.circle(arr, (int(round(x * 16)), int(round(y * 16))), int(round(r * 16)), tuple(int(c) for c in color),
               thick, cv2.LINE_AA, shift=4)  # fmt: skip


def fmt_fr(x: float, nd: int) -> str:
    """Nombre au format français : virgule décimale, espace fine insécable pour les milliers."""
    s = f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s.replace(" ", " ")
