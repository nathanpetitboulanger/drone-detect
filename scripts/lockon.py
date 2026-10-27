"""Logique de « verrouillage » (lock-on) d'une cible pour le guidage terminal par vision.

Module pur (ni OpenCV ni Ultralytics) : il reçoit à chaque frame la liste des pistes du
tracker et renvoie l'état du verrou et le vecteur d'erreur de visée. Il ne contient
**aucun code de contrôle de vol** : il s'arrête à l'erreur de visée, qui est l'entrée
qu'un contrôleur de guidage consommerait.

Entrée par frame
    Liste de pistes ``(track_id, x1, y1, x2, y2, conf, cls)`` en pixels (coin haut-gauche
    ``(x1, y1)``, coin bas-droit ``(x2, y2)``) + taille de l'image ``(W, H)``.

Machine à états
    ``SEARCHING``  pas de cible verrouillée ;
    ``LOCKED``     la cible verrouillée est présente dans les pistes de la frame ;
    ``COASTING``   la cible a disparu depuis 1 à ``coast_frames`` frames : on garde sa
                   dernière position connue ;
    puis retour à ``SEARCHING`` à la frame ``coast_frames + 1`` sans la cible.

Acquisition
    On verrouille la piste la plus confiante dont ``conf >= lock_conf`` et présente
    (avec ``conf >= lock_conf``) depuis au moins ``min_frames`` frames consécutives
    (évite de verrouiller un faux positif fugace). Égalité : plus petit ``track_id``.

Maintien
    Le verrou est conservé **tant que l'ID de la piste existe**, même si une autre piste
    est plus confiante ou si la confiance de la cible passe sous ``lock_conf``.
    Option ``reacquire_radius`` (désactivée par défaut) : en ``COASTING``, si une piste
    éligible apparaît à moins de ``reacquire_radius`` (en unités normalisées dx/dy) de la
    dernière position connue, on la verrouille aussitôt (le tracker a souvent changé d'ID).

Erreur de visée (convention)
    ``dx = (cx - W/2) / (W/2)`` et ``dy = (cy - H/2) / (H/2)``, bornés à [-1, 1], où
    ``(cx, cy)`` est le centre de la boîte en pixels :
    ``dx > 0`` : cible à **droite** du centre image, ``dx < 0`` : à gauche ;
    ``dy > 0`` : cible en **bas** (axe image vers le bas), ``dy < 0`` : en haut ;
    ``(0, 0)`` : cible centrée sur le réticule ; ``|dx| = 1`` : bord gauche/droit.
    Taille relative (proxy de distance) : ``box_area_ratio = (w * h) / (W * H)`` et
    ``box_size_ratio = sqrt(box_area_ratio)``. Pour une cible de taille physique fixe,
    ``box_size_ratio`` est à peu près inversement proportionnel à la distance.

Exemple
    >>> lock = LockOn(lock_conf=0.5, coast_frames=10)
    >>> out = lock.update([(7, 1100, 300, 1220, 380, 0.83, 4)], 1920, 1080)
    >>> out.state.value, out.target.track_id, round(out.aim.dx, 3), round(out.aim.dy, 3)
    ('LOCKED', 7, 0.208, -0.37)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence, Union


class LockState(str, Enum):
    """État du verrou."""

    SEARCHING = "SEARCHING"
    LOCKED = "LOCKED"
    COASTING = "COASTING"


@dataclass(frozen=True)
class Track:
    """Piste du tracker à une frame (boîte en pixels)."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int = 0

    @classmethod
    def from_tuple(cls, t: Union["Track", Sequence[float]]) -> "Track":
        """Construit une piste depuis ``(track_id, x1, y1, x2, y2, conf, cls)``."""
        if isinstance(t, Track):
            return t
        tid, x1, y1, x2, y2, conf, *rest = t
        return cls(int(tid), float(x1), float(y1), float(x2), float(y2), float(conf), int(rest[0]) if rest else 0)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class AimError:
    """Erreur de visée normalisée (voir la convention dans la docstring du module)."""

    dx: float
    dy: float
    box_area_ratio: float
    box_size_ratio: float


@dataclass(frozen=True)
class LockOutput:
    """Résultat d'une frame.

    ``target`` est la cible verrouillée (en ``COASTING`` : sa dernière observation),
    ``aim`` son erreur de visée ; tous deux valent ``None`` en ``SEARCHING``.
    ``lost_frames`` compte les frames consécutives sans la cible (0 si ``LOCKED``).
    ``new_lock`` vaut ``True`` à la frame où une cible vient d'être (ré)acquise.
    """

    state: LockState
    target: Track | None
    aim: AimError | None
    lost_frames: int
    n_tracks: int
    new_lock: bool = False


def aim_error(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> AimError:
    """Erreur de visée d'une boîte de centre ``(cx, cy)`` et de taille ``(w, h)`` en pixels.

    >>> e = aim_error(1920, 0, 10, 10, 1920, 1080)
    >>> e.dx, e.dy
    (1.0, -1.0)
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"taille d'image invalide : {img_w}x{img_h}")
    half_w, half_h = img_w / 2, img_h / 2
    dx = min(1.0, max(-1.0, (cx - half_w) / half_w))
    dy = min(1.0, max(-1.0, (cy - half_h) / half_h))
    area = min(1.0, max(0.0, (w * h) / (img_w * img_h)))
    return AimError(dx=dx, dy=dy, box_area_ratio=area, box_size_ratio=math.sqrt(area))


@dataclass
class LockOn:
    """Machine à états du verrouillage (une instance par flux vidéo).

    Paramètres
        lock_conf : confiance minimale pour acquérir une cible.
        coast_frames : nombre de frames pendant lesquelles on garde la dernière position
            d'une cible perdue avant de revenir en ``SEARCHING`` (0 = libération immédiate).
        min_frames : persistance minimale (frames consécutives avec ``conf >= lock_conf``)
            avant de pouvoir verrouiller une piste (1 = immédiat).
        reacquire_radius : rayon de ré-acquisition en ``COASTING`` en unités dx/dy
            normalisées (0 = désactivé).
    """

    lock_conf: float = 0.5
    coast_frames: int = 15
    min_frames: int = 1
    reacquire_radius: float = 0.0
    _target: Track | None = field(default=None, init=False, repr=False)
    _lost: int = field(default=0, init=False, repr=False)
    _streak: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.coast_frames < 0:
            raise ValueError("coast_frames doit être >= 0")
        if self.min_frames < 1:
            raise ValueError("min_frames doit être >= 1")

    @property
    def state(self) -> LockState:
        if self._target is None:
            return LockState.SEARCHING
        return LockState.LOCKED if self._lost == 0 else LockState.COASTING

    def reset(self) -> None:
        """Oublie la cible et l'historique des pistes (changement de vidéo, etc.)."""
        self._target, self._lost = None, 0
        self._streak.clear()

    def update(self, tracks: Iterable[Union[Track, Sequence[float]]], img_w: int, img_h: int) -> LockOutput:
        """Met à jour le verrou avec les pistes de la frame courante."""
        current = [Track.from_tuple(t) for t in tracks]
        by_id = {t.track_id: t for t in current}
        self._update_streaks(current)
        new_lock = False

        if self._target is not None:
            if self._target.track_id in by_id:
                # Verrou conservé tant que l'ID existe, quelle que soit la confiance.
                self._target, self._lost = by_id[self._target.track_id], 0
            else:
                self._lost += 1
                handover = self._reacquire_candidate(current, img_w, img_h)
                if handover is not None:
                    self._target, self._lost, new_lock = handover, 0, True
                elif self._lost > self.coast_frames:
                    self._target, self._lost = None, 0

        if self._target is None:
            best = self._best_candidate(current)
            if best is not None:
                self._target, self._lost, new_lock = best, 0, True

        aim = None
        if self._target is not None:
            t = self._target
            aim = aim_error(t.cx, t.cy, t.w, t.h, img_w, img_h)
        return LockOutput(self.state, self._target, aim, self._lost, len(current), new_lock)

    def _update_streaks(self, current: list[Track]) -> None:
        streak = {}
        for t in current:
            if t.conf >= self.lock_conf:
                streak[t.track_id] = self._streak.get(t.track_id, 0) + 1
        self._streak = streak

    def _eligible(self, current: list[Track]) -> list[Track]:
        return [t for t in current if t.conf >= self.lock_conf and self._streak.get(t.track_id, 0) >= self.min_frames]

    def _best_candidate(self, current: list[Track]) -> Track | None:
        eligible = self._eligible(current)
        return min(eligible, key=lambda t: (-t.conf, t.track_id)) if eligible else None

    def _reacquire_candidate(self, current: list[Track], img_w: int, img_h: int) -> Track | None:
        if self.reacquire_radius <= 0 or self._target is None:
            return None
        last = aim_error(self._target.cx, self._target.cy, 0, 0, img_w, img_h)
        near = []
        for t in self._eligible(current):
            e = aim_error(t.cx, t.cy, 0, 0, img_w, img_h)
            if math.hypot(e.dx - last.dx, e.dy - last.dy) <= self.reacquire_radius:
                near.append(t)
        return min(near, key=lambda t: (-t.conf, t.track_id)) if near else None
