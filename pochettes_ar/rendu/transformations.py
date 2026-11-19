"""Matrices 4x4 homogènes : translation, rotation, échelle, composition (M&S ch. 6).

Un point (x, y, z) s'écrit (x, y, z, 1) et une direction (x, y, z, 0) : une seule matrice
4x4 représente ainsi rotation, échelle et translation, et enchaîner des transformations
revient à multiplier leurs matrices.
"""

from functools import reduce

import numpy as np


def translation(tx, ty, tz):
    M = np.eye(4)
    M[:3, 3] = (tx, ty, tz)
    return M


def echelle(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0])


def rotation(axe, angle):
    """Rotation d'angle (radians) autour d'un axe passant par l'origine (formule de Rodrigues) :
    R = I + sin(a) [k]x + (1 - cos(a)) [k]x², avec [k]x la matrice du produit vectoriel par k."""
    k = np.asarray(axe, dtype=np.float64)
    k = k / np.linalg.norm(k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    M = np.eye(4)
    M[:3, :3] = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
    return M


def rotation_autour(point, axe, angle):
    """Rotation autour d'un axe passant par un point : on amène le point à l'origine,
    on tourne, on le ramène. (Les matrices s'appliquent de droite à gauche.)"""
    return compose(translation(*point), rotation(axe, angle), translation(*-np.asarray(point, dtype=np.float64)))


def compose(*matrices):
    """compose(A, B, C) = A @ B @ C : C est appliquée en premier."""
    return reduce(np.matmul, matrices)


def matrice_normales(M):
    """Les normales ne se transforment pas comme les points : pour rester perpendiculaires
    aux surfaces, elles suivent l'inverse transposée de la partie linéaire (M&S 6.2.2).
    Pour une rotation pure, c'est la rotation elle-même."""
    return np.linalg.inv(M[..., :3, :3]).swapaxes(-1, -2)
