"""Estimation de la pose de la pochette par rapport à la caméra (solvePnP).

Problème PnP (« Perspective-n-Point ») : on connaît n points 3D d'un objet dans son propre
repère et leurs projections 2D dans l'image ; on cherche la rotation R et la translation t
telles que, pour chaque point,
    pixel ~ K (R X_objet + t)            (modèle du sténopé, voir calibration.py).

Ici, les n = 4 points sont les coins de la pochette, détectés par reconnaissance.py.

Repère de la pochette, le même que celui des modèles 3D (voir outils/generer_modeles.py) :
    - unité = largeur de la pochette, origine au centre de la pochette ;
    - la pochette est dans le plan y = 0 ;
    - +X vers le bord droit, +Z vers le bord bas, +Y sort de la pochette (vers la caméra).
(X, Y, Z) est direct : Y = Z x X. Comme les modèles sont exprimés dans ce repère, la pose
trouvée ici est directement la transformation modèle -> caméra du rendu.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Pose:
    rvec: np.ndarray  # (3,) rotation sous forme axe x angle (Rodrigues), format OpenCV
    tvec: np.ndarray  # (3,) position de l'origine de la pochette dans le repère caméra
    R: np.ndarray  # (3, 3) matrice de rotation pochette -> caméra
    erreur: float  # erreur de reprojection moyenne des 4 coins, en pixels

    def matrice(self):
        """Matrice homogène 4x4 pochette -> caméra : X_camera = R X_pochette + t."""
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = self.R, self.tvec
        return M


def coins_pochette(aspect=1.0):
    """Coins 3D de la pochette, dans l'ordre des coins détectés :
    haut-gauche, haut-droit, bas-droit, bas-gauche (le haut est en z = -h/2)."""
    l, h = 0.5, 0.5 * aspect
    return np.array([[-l, 0, -h], [l, 0, -h], [l, 0, h], [-l, 0, h]], dtype=np.float64)


def estimer_pose(detection, calibration):
    """Pose de la pochette détectée, ou None si l'estimation échoue."""
    objet = coins_pochette(detection.aspect)
    image = detection.coins.astype(np.float64)

    # Méthode IPPE (Collins & Bartoli, 2014), spécialisée pour des points coplanaires.
    # Pour un plan, la pose se déduit de l'homographie plan -> image. IPPE la calcule de
    # façon analytique (sans initialisation), puis OpenCV l'affine en minimisant l'erreur
    # de reprojection (Levenberg-Marquardt).
    #
    # Ambiguïté des cibles planes : avec peu de perspective (pochette petite ou lointaine),
    # deux poses symétriques (pochette inclinée « vers l'avant » ou « vers l'arrière »)
    # donnent presque la même image. IPPE renvoie les deux ; solvePnPGeneric les trie par
    # erreur de reprojection et on garde la meilleure. Si l'écart est faible, le choix peut
    # alterner d'une image à l'autre : les axes « basculent » brièvement.
    nb, rvecs, tvecs, erreurs = cv2.solvePnPGeneric(
        objet, image, calibration.K, calibration.distorsion, flags=cv2.SOLVEPNP_IPPE
    )
    if nb == 0:
        return None
    rvec, tvec = rvecs[0].ravel(), tvecs[0].ravel()

    # La pochette doit être devant la caméra (z > 0 en convention OpenCV).
    if tvec[2] <= 0:
        return None

    # Rodrigues : vecteur rotation (axe unitaire x angle en radians) -> matrice 3x3.
    R, _ = cv2.Rodrigues(rvec)
    return Pose(rvec, tvec, R, float(erreurs[0].ravel()[0]))


def projeter(points, pose, K):
    """Projette des points 3D du repère pochette vers des pixels, avec le modèle du sténopé
    (sans distorsion), écrit à la main plutôt qu'avec cv2.projectPoints :
        1. passage dans le repère caméra : X_c = R X + t ;
        2. projection perspective : division par la profondeur Z_c ;
        3. passage en pixels avec K.
    Renvoie (pixels (n, 2), profondeurs Z_c (n,))."""
    camera = np.asarray(points, dtype=np.float64) @ pose.R.T + pose.tvec  # étape 1
    z = camera[:, 2]
    normalises = camera[:, :2] / z[:, None]  # étape 2 : (X/Z, Y/Z)
    pixels = normalises @ K[:2, :2].T + K[:2, 2]  # étape 3 : (fx X/Z + cx, fy Y/Z + cy)
    return pixels, z
