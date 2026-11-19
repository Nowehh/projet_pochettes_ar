"""Placage de textures : chargement et échantillonnage bilinéaire (M&S 11.4).

Coordonnées de texture (u, v) dans [0, 1] (répétées au-delà), v = 0 en bas de l'image.
Le texel (i, j) est un petit carré dont le centre est en ((i + 0.5) / largeur, ...).
"""

import cv2
import numpy as np


def charger_texture(chemin):
    """Image -> tableau float32 (h, l, 3) RGB dans [0, 1]. (Lecture du fichier par OpenCV.)"""
    image = cv2.imread(str(chemin), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(chemin)
    return np.ascontiguousarray(image[:, :, ::-1], dtype=np.float32) / 255.0


def echantillonner_bilineaire(texture, uv):
    """Couleurs (n, 3) de la texture aux coordonnées uv (n, 2).

    Au lieu de prendre le texel le plus proche (aspect « pixelisé » quand on s'approche), on
    mélange les 4 texels qui entourent le point, pondérés par la proximité :
        c = (1-a)(1-b) c00 + a(1-b) c10 + (1-a) b c01 + a b c11.
    """
    h, l = texture.shape[:2]
    # Position en « unités de texel », mesurée depuis le centre du premier texel.
    x = np.mod(uv[:, 0], 1.0) * l - 0.5
    y = (1.0 - np.mod(uv[:, 1], 1.0)) * h - 0.5  # v = 0 en bas, ligne 0 de l'image en haut
    x0, y0 = np.floor(x), np.floor(y)
    a, b = (x - x0)[:, None], (y - y0)[:, None]
    # Texture répétée : les voisins qui sortent de l'image sont pris de l'autre côté.
    x0, y0 = x0.astype(np.int64) % l, y0.astype(np.int64) % h
    x1, y1 = (x0 + 1) % l, (y0 + 1) % h
    return ((1 - a) * (1 - b) * texture[y0, x0] + a * (1 - b) * texture[y0, x1]
            + (1 - a) * b * texture[y1, x0] + a * b * texture[y1, x1])
