"""Lissage de la pose : réduit le bruit, suit les vrais mouvements, ignore les sauts isolés."""

import cv2
import numpy as np

from pochettes_ar.vision.lissage import (LisseurPoses, angle_entre, matrice_depuis_quaternion,
                                         quaternion_depuis_matrice)
from pochettes_ar.vision.pose import Pose
from tests.test_pose import DE_FACE, angle_entre as angle_matrices

ALEA = np.random.default_rng(1)


def pose(rotation_deg, t):
    R = cv2.Rodrigues(np.radians(rotation_deg))[0] @ DE_FACE
    return Pose(cv2.Rodrigues(R)[0].ravel(), np.array(t, dtype=np.float64), R, 0.5)


def test_quaternions_aller_retour():
    for _ in range(50):
        R = cv2.Rodrigues(ALEA.uniform(-np.pi, np.pi, 3))[0]
        q = quaternion_depuis_matrice(R)
        np.testing.assert_allclose(np.linalg.norm(q), 1)
        np.testing.assert_allclose(matrice_depuis_quaternion(q), R, atol=1e-9)
    q1 = quaternion_depuis_matrice(cv2.Rodrigues(np.array([0.0, 0.0, 0.3]))[0])
    q2 = quaternion_depuis_matrice(cv2.Rodrigues(np.array([0.0, 0.0, 0.8]))[0])
    np.testing.assert_allclose(angle_entre(q1, q2), 0.5)
    np.testing.assert_allclose(angle_entre(q1, -q2), 0.5)  # q et -q : même rotation


def test_lissage_reduit_le_bruit():
    lisseur = LisseurPoses()
    vraie = pose([20, 10, 5], [0.1, -0.05, 2.5])
    brut, lisse = [], []
    for i in range(200):
        mesure = pose([20, 10, 5] + ALEA.normal(0, 0.8, 3), [0.1, -0.05, 2.5] + ALEA.normal(0, 0.01, 3))
        sortie = lisseur.lisser("a", mesure, i / 30)
        if i > 20:
            brut.append(angle_matrices(mesure.R, vraie.R))
            lisse.append(angle_matrices(sortie.R, vraie.R))
    assert np.mean(lisse) < 0.6 * np.mean(brut)


def test_suit_un_vrai_mouvement():
    """Une pochette qui tourne à vitesse constante : la pose lissée ne traîne pas loin derrière."""
    lisseur = LisseurPoses()
    for i in range(60):
        mesure = pose([0, 2 * i, 0], [0, 0, 2.5])  # 2° par image, 60°/s à 30 i/s
        sortie = lisseur.lisser("a", mesure, i / 30)
    assert angle_matrices(sortie.R, mesure.R) < 2.0


def test_saut_isole_ignore_puis_confirme():
    lisseur = LisseurPoses(confirmation=3)
    stable, bascule = pose([30, 0, 0], [0, 0, 2.5]), pose([-30, 0, 0], [0, 0, 2.5])
    for i in range(10):
        lisseur.lisser("a", stable, i / 30)
    # une image basculée : ignorée
    sortie = lisseur.lisser("a", bascule, 10 / 30)
    assert angle_matrices(sortie.R, stable.R) < 1
    assert angle_matrices(lisseur.lisser("a", stable, 11 / 30).R, stable.R) < 1
    # trois images basculées d'affilée : adoptée
    for i in range(12, 15):
        sortie = lisseur.lisser("a", bascule, i / 30)
    assert angle_matrices(sortie.R, bascule.R) < 1


def test_oubli_et_albums_independants():
    lisseur = LisseurPoses(oubli=0.5)
    a, b = pose([10, 0, 0], [0, 0, 2]), pose([-20, 40, 0], [0.5, 0, 3])
    lisseur.lisser("a", a, 0.0)
    assert lisseur.lisser("b", b, 0.0) is b  # premier passage : pose inchangée
    autre = pose([12, 0, 0], [0.3, 0, 2])
    sortie = lisseur.lisser("a", autre, 2.0)  # « a » perdue depuis 2 s : on repart de zéro
    np.testing.assert_allclose(sortie.tvec, autre.tvec)
