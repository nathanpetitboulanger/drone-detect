"""Tests unitaires de la logique de verrouillage (scripts/lockon.py).

Usage : ``.venv/bin/python -m pytest tests/ -q``
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lockon import LockOn, LockState, Track, aim_error  # noqa: E402

W, H = 1920, 1080


def box(tid, cx, cy, conf, size=40, cls=0):
    """Piste carrée de côté ``size`` centrée en ``(cx, cy)``."""
    s = size / 2
    return (tid, cx - s, cy - s, cx + s, cy + s, conf, cls)


# --- Acquisition -----------------------------------------------------------------------


def test_searching_without_tracks():
    out = LockOn().update([], W, H)
    assert out.state is LockState.SEARCHING
    assert out.target is None and out.aim is None
    assert out.n_tracks == 0 and not out.new_lock


def test_acquires_most_confident_track():
    lock = LockOn(lock_conf=0.5)
    out = lock.update([box(1, 100, 100, 0.6), box(2, 500, 500, 0.9), box(3, 900, 900, 0.4)], W, H)
    assert out.state is LockState.LOCKED
    assert out.target.track_id == 2
    assert out.new_lock and out.lost_frames == 0 and out.n_tracks == 3


def test_no_lock_below_lock_conf():
    out = LockOn(lock_conf=0.5).update([box(1, 100, 100, 0.49)], W, H)
    assert out.state is LockState.SEARCHING and out.target is None


def test_tie_broken_by_lowest_id():
    out = LockOn().update([box(9, 100, 100, 0.8), box(4, 500, 500, 0.8)], W, H)
    assert out.target.track_id == 4


def test_min_frames_persistence():
    lock = LockOn(lock_conf=0.5, min_frames=3)
    assert lock.update([box(1, 100, 100, 0.9)], W, H).state is LockState.SEARCHING
    assert lock.update([box(1, 100, 100, 0.9)], W, H).state is LockState.SEARCHING
    out = lock.update([box(1, 100, 100, 0.9)], W, H)
    assert out.state is LockState.LOCKED and out.target.track_id == 1 and out.new_lock


def test_min_frames_streak_reset_by_gap_or_low_conf():
    lock = LockOn(lock_conf=0.5, min_frames=2)
    lock.update([box(1, 100, 100, 0.9)], W, H)
    lock.update([], W, H)  # trou : la série repart de zéro
    assert lock.update([box(1, 100, 100, 0.9)], W, H).state is LockState.SEARCHING
    lock.update([box(1, 100, 100, 0.3)], W, H)  # sous lock_conf : série remise à zéro
    assert lock.update([box(1, 100, 100, 0.9)], W, H).state is LockState.SEARCHING
    assert lock.update([box(1, 100, 100, 0.9)], W, H).state is LockState.LOCKED


# --- Maintien du verrou ----------------------------------------------------------------


def test_lock_kept_against_more_confident_track():
    lock = LockOn(lock_conf=0.5)
    assert lock.update([box(1, 100, 100, 0.6)], W, H).target.track_id == 1
    out = lock.update([box(1, 110, 100, 0.55), box(2, 500, 500, 0.95)], W, H)
    assert out.state is LockState.LOCKED and out.target.track_id == 1 and not out.new_lock
    # Même sous lock_conf, l'ID existe toujours : le verrou tient.
    out = lock.update([box(1, 120, 100, 0.2), box(2, 500, 500, 0.95)], W, H)
    assert out.state is LockState.LOCKED and out.target.track_id == 1
    assert out.target.cx == pytest.approx(120)


def test_target_position_follows_track():
    lock = LockOn()
    lock.update([box(1, 960, 540, 0.9)], W, H)
    out = lock.update([box(1, 1440, 270, 0.9)], W, H)
    assert out.aim.dx == pytest.approx(0.5) and out.aim.dy == pytest.approx(-0.5)


# --- Coasting et libération ------------------------------------------------------------


def test_coasting_then_release_to_searching():
    lock = LockOn(lock_conf=0.5, coast_frames=3)
    first = lock.update([box(1, 1440, 810, 0.9)], W, H)
    for k in range(1, 4):
        out = lock.update([], W, H)
        assert out.state is LockState.COASTING
        assert out.lost_frames == k
        assert out.target.track_id == 1  # dernière position conservée
        assert out.aim == first.aim
    out = lock.update([], W, H)
    assert out.state is LockState.SEARCHING
    assert out.target is None and out.aim is None and out.lost_frames == 0


def test_coasting_ignores_other_tracks_then_switches_after_release():
    lock = LockOn(lock_conf=0.5, coast_frames=2)
    lock.update([box(1, 100, 100, 0.7)], W, H)
    for _ in range(2):
        out = lock.update([box(2, 900, 900, 0.95)], W, H)
        assert out.state is LockState.COASTING and out.target.track_id == 1
    out = lock.update([box(2, 900, 900, 0.95)], W, H)
    assert out.state is LockState.LOCKED and out.target.track_id == 2 and out.new_lock


def test_reappearing_target_during_coasting_is_relocked():
    lock = LockOn(coast_frames=5)
    lock.update([box(1, 100, 100, 0.9)], W, H)
    lock.update([], W, H)
    lock.update([], W, H)
    out = lock.update([box(1, 130, 100, 0.9), box(2, 900, 900, 0.99)], W, H)
    assert out.state is LockState.LOCKED and out.target.track_id == 1
    assert out.lost_frames == 0 and not out.new_lock


def test_coast_frames_zero_releases_immediately():
    lock = LockOn(coast_frames=0)
    lock.update([box(1, 100, 100, 0.9)], W, H)
    assert lock.update([], W, H).state is LockState.SEARCHING


def test_reacquire_radius_handover_near_last_position():
    lock = LockOn(coast_frames=10, reacquire_radius=0.1)
    lock.update([box(1, 960, 540, 0.9)], W, H)
    out = lock.update([box(5, 1000, 540, 0.8)], W, H)  # dx = 0.042 : assez proche
    assert out.state is LockState.LOCKED and out.target.track_id == 5 and out.new_lock


def test_reacquire_radius_ignores_far_tracks():
    lock = LockOn(coast_frames=10, reacquire_radius=0.1)
    lock.update([box(1, 960, 540, 0.9)], W, H)
    out = lock.update([box(6, 1800, 540, 0.99)], W, H)  # dx = 0.875 : trop loin
    assert out.state is LockState.COASTING and out.target.track_id == 1


def test_reset_forgets_target():
    lock = LockOn()
    lock.update([box(1, 100, 100, 0.9)], W, H)
    lock.reset()
    assert lock.state is LockState.SEARCHING
    assert lock.update([], W, H).target is None


# --- Erreur de visée -------------------------------------------------------------------


def test_aim_error_center_is_zero():
    e = aim_error(W / 2, H / 2, 10, 10, W, H)
    assert e.dx == pytest.approx(0) and e.dy == pytest.approx(0)


@pytest.mark.parametrize(
    "cx, cy, dx, dy",
    [
        (0, 0, -1, -1),  # coin haut-gauche
        (W, 0, 1, -1),  # coin haut-droit
        (0, H, -1, 1),  # coin bas-gauche
        (W, H, 1, 1),  # coin bas-droit
        (W * 0.75, H / 2, 0.5, 0),  # à droite : dx > 0
        (W / 2, H * 0.75, 0, 0.5),  # en bas : dy > 0
    ],
)
def test_aim_error_corners_and_signs(cx, cy, dx, dy):
    e = aim_error(cx, cy, 10, 10, W, H)
    assert e.dx == pytest.approx(dx) and e.dy == pytest.approx(dy)


def test_aim_error_non_square_image():
    # 1280x720 : demi-largeur 640, demi-hauteur 360 (normalisation par axe).
    e = aim_error(960, 180, 128, 72, 1280, 720)
    assert e.dx == pytest.approx(0.5) and e.dy == pytest.approx(-0.5)
    assert e.box_area_ratio == pytest.approx(0.01)
    assert e.box_size_ratio == pytest.approx(0.1)
    # Portrait 720x1280 : même point relatif.
    e = aim_error(540, 320, 10, 10, 720, 1280)
    assert e.dx == pytest.approx(0.5) and e.dy == pytest.approx(-0.5)


def test_aim_error_clipped_outside_image():
    e = aim_error(-100, H + 50, 10, 10, W, H)
    assert (e.dx, e.dy) == (-1.0, 1.0)


def test_aim_error_invalid_image_size():
    with pytest.raises(ValueError):
        aim_error(0, 0, 1, 1, 0, 720)


def test_locked_output_uses_box_center_and_size():
    out = LockOn().update([(3, 1280, 270, 1440, 360, 0.8, 4)], W, H)
    t = out.target
    assert (t.cx, t.cy, t.w, t.h, t.cls) == (1360, 315, 160, 90, 4)
    assert out.aim.dx == pytest.approx((1360 - 960) / 960)
    assert out.aim.dy == pytest.approx((315 - 540) / 540)
    assert out.aim.box_area_ratio == pytest.approx(160 * 90 / (W * H))


def test_track_from_tuple_without_cls():
    t = Track.from_tuple((7, 0, 0, 10, 20, 0.5))
    assert t.cls == 0 and t.w == 10 and t.h == 20
