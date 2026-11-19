"""Pose : on fabrique des vues à partir d'une pose connue et on vérifie qu'on la retrouve."""

import cv2
import numpy as np
import pytest

from pochettes_ar.vision import calibration
from pochettes_ar.vision.pose import coins_pochette, estimer_pose, projeter
from pochettes_ar.vision.reconnaissance import Detection
from tests import test_reconnaissance as T

CALIB = calibration.approximer(T.LARGEUR, T.HAUTEUR)

# Pochette vue de face : X du repère pochette -> x caméra (droite), Z -> y caméra (bas),
# Y (normale sortante) -> -z caméra (vers la caméra).
DE_FACE = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

POSES = {
    "face": (np.zeros(3), [0.0, 0.0, 2.5]),
    "inclinee": (np.radians([35, 0, 0]), [0.2, -0.1, 2.2]),
    "tournee": (np.radians([-20, 30, 15]), [-0.3, 0.1, 2.8]),
    "lointaine": (np.radians([10, -25, 5]), [0.6, 0.3, 5.0]),
}


def pose_vraie(nom):
    rotation, t = POSES[nom]
    R = cv2.Rodrigues(rotation)[0] @ DE_FACE
    return R, np.array(t)


def angle_entre(R1, R2):
    """Angle (degrés) de la rotation qui fait passer de R1 à R2."""
    return np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


def projection_opencv(R, t, points):
    return cv2.projectPoints(points, cv2.Rodrigues(R)[0], t, CALIB.K, CALIB.distorsion)[0].reshape(-1, 2)


def detection(coins):
    return Detection("x", "x", np.float32(coins), np.eye(3), 0, 0)


def test_projeter_comme_opencv():
    R, t = pose_vraie("tournee")
    pose = estimer_pose(detection(projection_opencv(R, t, coins_pochette())), CALIB)
    points = np.random.default_rng(0).uniform(-1, 1, (50, 3))
    nos_pixels, _ = projeter(points, pose, CALIB.K)
    np.testing.assert_allclose(nos_pixels, projection_opencv(pose.R, pose.tvec, points), atol=1e-6)


@pytest.mark.parametrize("nom", POSES)
def test_retrouve_la_pose_exacte(nom):
    """Coins parfaits (aux arrondis float32 près) : la pose doit être retrouvée quasi exactement."""
    R, t = pose_vraie(nom)
    pose = estimer_pose(detection(projection_opencv(R, t, coins_pochette())), CALIB)
    assert angle_entre(pose.R, R) < 0.1
    np.testing.assert_allclose(pose.tvec, t, atol=1e-3)
    assert pose.erreur < 0.05


def test_axes_de_face():
    """Pochette de face, centrée : X vers la droite de l'image, Z vers le bas, Y vers la caméra."""
    c = np.array([T.LARGEUR / 2, T.HAUTEUR / 2])
    coins = c + 200 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    pose = estimer_pose(detection(coins), CALIB)
    np.testing.assert_allclose(pose.R, DE_FACE, atol=1e-6)
    assert pose.tvec[2] > 0  # devant la caméra


@pytest.mark.parametrize("nom", POSES)
def test_bout_en_bout(nom):
    """Scène synthétique -> reconnaissance -> pose : comparée à la pose qui a servi à la fabriquer."""
    R, t = pose_vraie(nom)
    rng = np.random.default_rng(42)
    scene = T.fond(rng)
    T.incruster(scene, "radiohead_ok_computer", projection_opencv(R, t, coins_pochette()), 0.8)
    detections = T.Reconnaisseur(T.ALBUMS).reconnaitre(T.degrader(scene, rng))
    assert [d.id for d in detections] == ["radiohead_ok_computer"]

    pose = estimer_pose(detections[0], CALIB)
    assert angle_entre(pose.R, R) < 3  # degrés
    assert np.linalg.norm(pose.tvec - t) < 0.02 * np.linalg.norm(t)
