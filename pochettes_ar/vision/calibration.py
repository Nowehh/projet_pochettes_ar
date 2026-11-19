"""Paramètres intrinsèques de la caméra : matrice K et coefficients de distorsion.

Modèle du sténopé (M&S ch. 7) : un point (X, Y, Z) du repère caméra (convention OpenCV :
x à droite, y vers le bas, z vers l'avant) se projette au pixel
    u = fx X / Z + cx,   v = fy Y / Z + cy,
soit, en coordonnées homogènes, p ~ K (X, Y, Z) avec
    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]].
fx, fy : focale exprimée en pixels ; (cx, cy) : point principal (où l'axe optique perce l'image).

Si calibration/camera.json existe (issu d'une vraie calibration), on l'utilise. Sinon on
approxime K : pixels carrés (fx = fy), point principal au centre de l'image, pas de distorsion,
et une focale déduite d'un champ de vision horizontal supposé.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Champ de vision horizontal supposé quand on n'a pas de calibration. Les webcams d'ordinateur
# portable sont typiquement entre 55° et 70°. Une erreur sur cette valeur fausse surtout la
# distance estimée et l'inclinaison, pas la position de la pochette dans l'image.
CHAMP_HORIZONTAL_DEFAUT = 60.0  # degrés

FICHIER = Path(__file__).resolve().parents[2] / "calibration" / "camera.json"


@dataclass
class Calibration:
    K: np.ndarray  # (3, 3)
    distorsion: np.ndarray  # (5,) coefficients k1, k2, p1, p2, k3 (convention OpenCV)
    approximee: bool  # vrai si K vient de approximer() et non d'une calibration


def approximer(largeur, hauteur, champ_horizontal=CHAMP_HORIZONTAL_DEFAUT):
    """K approximée : la demi-largeur de l'image est vue sous un angle champ/2, donc
    tan(champ / 2) = (largeur / 2) / f   =>   f = (largeur / 2) / tan(champ / 2)."""
    f = (largeur / 2) / np.tan(np.radians(champ_horizontal) / 2)
    K = np.array([[f, 0, largeur / 2], [0, f, hauteur / 2], [0, 0, 1]], dtype=np.float64)
    return Calibration(K, np.zeros(5), approximee=True)


def charger(largeur, hauteur, fichier=FICHIER):
    """Calibration pour des images de taille largeur x hauteur.

    Le fichier JSON contient {"largeur", "hauteur", "K", "distorsion"}. Si la résolution
    diffère de celle de la calibration (même capteur, même cadrage), on met K à l'échelle.
    """
    fichier = Path(fichier)
    if not fichier.exists():
        return approximer(largeur, hauteur)
    donnees = json.loads(fichier.read_text())
    K = np.array(donnees["K"], dtype=np.float64)
    K[0] *= largeur / donnees["largeur"]  # fx, cx
    K[1] *= hauteur / donnees["hauteur"]  # fy, cy
    return Calibration(K, np.array(donnees["distorsion"], dtype=np.float64), approximee=False)
