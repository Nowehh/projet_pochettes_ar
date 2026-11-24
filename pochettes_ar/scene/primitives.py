"""Primitives géométriques (sphère, disque, cylindre, cône, pavé, tube…) construites par le code.

Chaque primitive est une surface paramétrée P(u, v) échantillonnée sur une grille, avec ses
normales et ses coordonnées de texture (M&S 2.5 et 12.1). Elles servent à la fois aux modèles
faits à la main (outils/generer_modeles.py) et aux modèles composés par IA à partir d'une
description de la pochette (pochettes_ar/generation/construction.py).
"""

import numpy as np

from pochettes_ar.rendu.transformations import translation
from pochettes_ar.scene.maillage import Maillage

# --- Placement ------------------------------------------------------------------------


def transformer(m, matrice):
    """Applique une matrice 4x4 ; les normales suivent l'inverse transposée (M&S 6.2.2)."""
    lin = matrice[:3, :3]
    sommets = m.sommets @ lin.T + matrice[:3, 3]
    normales = m.normales @ np.linalg.inv(lin)  # = (inv(lin).T @ n.T).T
    normales /= np.linalg.norm(normales, axis=1, keepdims=True)
    triangles = m.triangles if np.linalg.det(lin) > 0 else m.triangles[:, ::-1]
    return Maillage(sommets, normales, m.uv.copy(), triangles)


# --- Primitives -----------------------------------------------------------------------


def grille(P, N):
    """Surface paramétrée échantillonnée : P, N de forme (nu+1, nv+1, 3), u = premier indice.

    La normale sortante doit être du côté de dP/du x dP/dv. Les UV vont de 0 à 1.
    Les triangles dégénérés (pôles d'une sphère) sont retirés.
    """
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    u, v = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), indexing="ij")
    idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    triangles = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])

    sommets = P.reshape(-1, 3)
    t = sommets[triangles]
    aires = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    return Maillage(sommets, N.reshape(-1, 3), np.stack([u.ravel(), v.ravel()], 1), triangles[aires > 1e-12])


def sphere(lat=(-90, 90), lon=(0, 360), nu=48, nv=24):
    """Morceau de sphère unité. lon = 0 pointe vers +Z, lon = 90 vers +X (degrés)."""
    lo, la = np.meshgrid(np.radians(np.linspace(*lon, nu + 1)), np.radians(np.linspace(*lat, nv + 1)), indexing="ij")
    P = np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], -1)
    return grille(P, P.copy())


def disque(n=48, haut=True):
    """Disque unité dans le plan y = 0, normale +Y (haut=True) ou -Y. UV planaires."""
    angles = np.radians(np.linspace(0, 360, n + 1))[:-1]
    anneau = np.stack([np.sin(angles), np.zeros(n), np.cos(angles)], 1)
    sommets = np.concatenate([[[0.0, 0.0, 0.0]], anneau])
    uv = 0.5 + 0.5 * sommets[:, [0, 2]] * [1, -1]
    k = np.arange(1, n + 1)
    triangles = np.stack([np.zeros(n, int), k, k % n + 1], 1)
    normale = [0.0, 1.0, 0.0]
    if not haut:
        triangles, normale = triangles[:, ::-1], [0.0, -1.0, 0.0]
    return Maillage(sommets, np.tile(normale, (n + 1, 1)), uv, triangles)


def cylindre(n=32, bouchons=True):
    """Cylindre de rayon 1, axe Y, de y = 0 à y = 1."""
    an, y = np.meshgrid(np.radians(np.linspace(0, 360, n + 1)), [0.0, 1.0], indexing="ij")
    N = np.stack([np.sin(an), np.zeros_like(an), np.cos(an)], -1)
    P = N + np.stack([np.zeros_like(y), y, np.zeros_like(y)], -1)
    morceaux = [grille(P, N)]
    if bouchons:
        morceaux += [disque(n, haut=False), transformer(disque(n), translation(0, 1, 0))]
    return Maillage.fusion(morceaux)


def quad(origine, e1, e2, nu=1, nv=1):
    """Parallélogramme origine + u e1 + v e2 ; normale e1 x e2."""
    origine, e1, e2 = (np.asarray(x, dtype=float) for x in (origine, e1, e2))
    u, v = np.meshgrid(np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1), indexing="ij")
    P = origine + u[..., None] * e1 + v[..., None] * e2
    n = np.cross(e1, e2)
    return grille(P, np.broadcast_to(n / np.linalg.norm(n), P.shape).copy())


def boite(sx, sy, sz, centre=(0, 0, 0)):
    """Pavé centré, normales par face (arêtes vives). Chaque face a ses UV dans [0, 1]."""
    a, b, c = sx / 2, sy / 2, sz / 2
    faces = [
        quad((a, -b, c), (0, 0, -2 * c), (0, 2 * b, 0)),  # +X
        quad((-a, -b, -c), (0, 0, 2 * c), (0, 2 * b, 0)),  # -X
        quad((-a, b, c), (2 * a, 0, 0), (0, 0, -2 * c)),  # +Y
        quad((-a, -b, -c), (2 * a, 0, 0), (0, 0, 2 * c)),  # -Y
        quad((-a, -b, c), (2 * a, 0, 0), (0, 2 * b, 0)),  # +Z
        quad((a, -b, -c), (-2 * a, 0, 0), (0, 2 * b, 0)),  # -Z
    ]
    return transformer(Maillage.fusion(faces), translation(*centre))


def tube(points, rayon, n=12):
    """Tube (sans bouchons) le long d'une polyligne, repère par transport parallèle."""
    points = np.asarray(points, dtype=float)
    tangentes = np.gradient(points, axis=0)
    tangentes /= np.linalg.norm(tangentes, axis=1, keepdims=True)
    reference = np.array([1.0, 0, 0]) if abs(tangentes[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    normal = np.cross(tangentes[0], reference)
    reperes = []
    for t in tangentes:
        normal = normal - np.dot(normal, t) * t
        normal /= np.linalg.norm(normal)
        reperes.append((normal, np.cross(t, normal)))
    angles = np.radians(np.linspace(0, 360, n + 1))
    # u = angle autour du tube, v = abscisse le long du tube ; dP/du x dP/dv est sortant.
    N = np.array([[np.cos(a) * nn + np.sin(a) * bb for nn, bb in reperes] for a in angles])
    return grille(points[None] + rayon * N, N)


def cone(n=32, bouchon=True):
    """Cône de rayon de base 1, axe Y, base en y = 0, pointe en y = 1.

    Normale de la surface latérale : pour x² + z² = (1 - y)², le gradient de
    f = sqrt(x² + z²) + y - 1 donne n ∝ (sin a, 1, cos a). Les sommets de la pointe sont
    dupliqués (un par angle) pour garder une normale lisse autour."""
    an, y = np.meshgrid(np.radians(np.linspace(0, 360, n + 1)), [0.0, 1.0], indexing="ij")
    r = 1.0 - y
    P = np.stack([r * np.sin(an), y, r * np.cos(an)], -1)
    N = np.stack([np.sin(an), np.ones_like(an), np.cos(an)], -1) / np.sqrt(2)
    # grille() retire les triangles dégénérés de la pointe : il reste un triangle par secteur.
    morceaux = [grille(P, N)]
    if bouchon:
        morceaux.append(disque(n, haut=False))
    return Maillage.fusion(morceaux)
