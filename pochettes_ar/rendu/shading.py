"""Éclairage de Blinn-Phong (M&S 5.2 et 5.3), calculé par pixel (« Phong shading » : la
normale est interpolée puis renormalisée en chaque pixel, M&S 10.2).

Pour une source directionnelle :
    L = ka kd  +  I kd max(0, n.l)  +  I ks max(0, n.h)^p
- n : normale unitaire ; l : direction vers la lumière ; v : direction vers l'œil ;
- h = (l + v) / |l + v| : « vecteur à mi-chemin ». Il est proche de n quand la direction de
  réflexion miroir de l est proche de v, d'où le reflet ;
- kd : couleur diffuse (texture x Kd du matériau) ; ks : couleur spéculaire ;
- p (Ns du .mtl) : exposant de Phong. Plus il est grand, plus le reflet est petit et net ;
- ka : lumière ambiante, terme constant qui évite que les faces non éclairées soient noires.
"""

import numpy as np


def normaliser(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def blinn_phong(n, l, v, kd, ks, p, ambiante=0.3, intensite=1.0):
    """Tous les vecteurs sont (n, 3) (ou (3,) pour l), dans le même repère ; p est (n,)."""
    n_l = np.einsum("ij,ij->i", n, np.broadcast_to(l, n.shape))
    h = normaliser(l + v)
    n_h = np.maximum(np.einsum("ij,ij->i", n, h), 0.0)
    # Pas de reflet sur une face qui tourne le dos à la lumière.
    speculaire = np.where(n_l > 0, n_h ** p, 0.0)[:, None]
    diffus = np.maximum(n_l, 0.0)[:, None]
    return np.clip(ambiante * kd + intensite * (kd * diffus + ks * speculaire), 0.0, 1.0)
