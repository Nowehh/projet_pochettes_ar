"""Viewing : projection perspective construite depuis K, et viewport (M&S ch. 7).

Chaîne complète d'un sommet, d'après M&S (7.1 et 7.3), adaptée à la convention OpenCV :

    modèle --M_modele--> pochette --M_vue--> caméra --P--> clip --/w--> NDC --M_viewport--> pixels

- M_modele : place la partie du modèle sur la pochette (animation) ;
- M_vue    : la pose estimée par solvePnP (pochette -> caméra) ;
- P        : projection perspective, construite à partir de K (cette page) ;
- /w       : division perspective (c'est elle qui rend les objets lointains plus petits) ;
- M_viewport : du volume canonique [-1, 1]³ (NDC) vers les pixels de l'image.

Repère caméra OpenCV : x à droite, y vers le bas, z vers l'avant (le livre regarde vers -z
avec y vers le haut). On garde la convention OpenCV jusqu'au bout, pour que K et la pose de
solvePnP s'appliquent sans conversion : en NDC aussi, y descend.
"""

import numpy as np


def matrice_projection(K, largeur, hauteur, pres, loin):
    """Matrice 4x4 P : caméra -> coordonnées de clip, avec w_clip = Z (profondeur).

    Après division par w, on obtient :
      x_ndc = 2 (fx X/Z + cx + 0.5) / largeur - 1   (le pixel de K, ramené dans [-1, 1])
      y_ndc = 2 (fy Y/Z + cy + 0.5) / hauteur - 1
      z_ndc = A + B / Z, avec A et B choisis pour que Z = pres donne -1 et Z = loin donne +1.
    Le +0.5 vient de la convention des pixels : chez OpenCV, le centre du pixel (0, 0) est à la
    coordonnée 0, donc l'image couvre [-0.5, largeur - 0.5] (même convention que M&S 3.2).

    z_ndc croît avec la distance : le z-buffer garde la plus petite valeur. C'est une fonction
    de 1/Z, plus précise près de la caméra. Comme 1/Z varie linéairement à l'écran sur un
    triangle, z_ndc peut s'interpoler linéairement avec les barycentriques de l'écran.
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    A = (loin + pres) / (loin - pres)
    B = -2 * loin * pres / (loin - pres)
    return np.array(
        [
            [2 * fx / largeur, 0, 2 * (cx + 0.5) / largeur - 1, 0],
            [0, 2 * fy / hauteur, 2 * (cy + 0.5) / hauteur - 1, 0],
            [0, 0, A, B],
            [0, 0, 1, 0],  # w_clip = Z
        ]
    )


def matrice_viewport(largeur, hauteur):
    """NDC [-1, 1]² -> pixels [-0.5, largeur - 0.5] x [-0.5, hauteur - 0.5] (M&S éq. 7.2) :
    x_pixel = (largeur / 2) x_ndc + (largeur - 1) / 2. La profondeur z_ndc est conservée."""
    return np.array(
        [
            [largeur / 2, 0, 0, (largeur - 1) / 2],
            [0, hauteur / 2, 0, (hauteur - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
