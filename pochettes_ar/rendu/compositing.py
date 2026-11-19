"""Incrustation de l'image rendue sur l'image webcam.

Le masque vient du tampon des triangles : là où un triangle est visible, on remplace le
pixel de la webcam par la couleur calculée, ailleurs on garde la webcam. (Pas d'anticrénelage :
les bords de l'objet sont en marches d'escalier.)
"""

import numpy as np


def composer(fond_bgr, ys, xs, couleurs_rgb):
    """Copie de fond_bgr (uint8, BGR comme OpenCV) avec les pixels (ys, xs) remplacés par
    couleurs_rgb (float dans [0, 1], RGB comme nos textures et matériaux)."""
    image = fond_bgr.copy()
    image[ys, xs] = (couleurs_rgb[:, ::-1] * 255.0 + 0.5).astype(np.uint8)
    return image
