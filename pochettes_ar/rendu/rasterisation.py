"""Rasterisation des triangles : coordonnées barycentriques, test de profondeur et
interpolation corrigée de la perspective (M&S 8.1).

Ce module contient la seule boucle « pixel par pixel » du moteur. Elle est écrite en
Python et compilée en code machine par Numba (@njit) : l'algorithme est le nôtre, Numba ne
fait que l'accélérer. Sans Numba, ces boucles imbriquées prendraient plusieurs secondes
par image.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _rasteriser(ecran, prof, inv_w, id_depart, profondeur, triangle, bary):
    hauteur, largeur = profondeur.shape
    for t in range(ecran.shape[0]):
        x0, y0 = ecran[t, 0, 0], ecran[t, 0, 1]
        x1, y1 = ecran[t, 1, 0], ecran[t, 1, 1]
        x2, y2 = ecran[t, 2, 0], ecran[t, 2, 1]

        # Aire signée (x2) du triangle à l'écran. Nulle : triangle vu par la tranche, rien à dessiner.
        aire = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if aire == 0.0:
            continue

        # 1. Boîte englobante : seuls les pixels dont le centre (coordonnées entières) est
        #    dans [min, max] peuvent être couverts. On la coupe aux bords de l'image : c'est
        #    notre « clipping » écran, pixel par pixel.
        xmin = max(int(np.ceil(min(x0, x1, x2))), 0)
        xmax = min(int(np.floor(max(x0, x1, x2))), largeur - 1)
        ymin = max(int(np.ceil(min(y0, y1, y2))), 0)
        ymax = min(int(np.floor(max(y0, y1, y2))), hauteur - 1)

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                # 2. Coordonnées barycentriques du centre du pixel p = (x, y) (M&S 2.7) :
                #    l0 = aire(p, p1, p2) / aire(p0, p1, p2), etc. Elles valent 1 au sommet
                #    concerné, 0 sur le côté opposé, et sont toutes >= 0 dans le triangle.
                l0 = ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) / aire
                l1 = ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y)) / aire
                l2 = 1.0 - l0 - l1
                if l0 < 0.0 or l1 < 0.0 or l2 < 0.0:
                    continue  # pixel hors du triangle
                # (Un pixel exactement sur une arête commune à deux triangles est accepté par
                # les deux. Sans transparence, ce n'est pas gênant : le z-buffer tranche.)

                # 3. Test de profondeur (z-buffer). z_ndc est affine à l'écran (voir camera.py) :
                #    on peut l'interpoler directement avec l0, l1, l2.
                z = l0 * prof[t, 0] + l1 * prof[t, 1] + l2 * prof[t, 2]
                if z < -1.0 or z > 1.0 or z >= profondeur[y, x]:
                    continue  # hors de [pres, loin], ou caché par un fragment plus proche
                profondeur[y, x] = z
                triangle[y, x] = id_depart + t

                # 4. Barycentriques corrigées de la perspective (M&S 11.2.4). Un attribut
                #    (UV, normale, position) n'est PAS affine à l'écran, mais attribut / w
                #    et 1 / w le sont. On interpole donc li / wi puis on renormalise :
                #        bi = (li / wi) / (l0 / w0 + l1 / w1 + l2 / w2).
                #    Sans cette correction, les textures « glissent » sur les triangles inclinés.
                q0 = l0 * inv_w[t, 0]
                q1 = l1 * inv_w[t, 1]
                q2 = l2 * inv_w[t, 2]
                s = q0 + q1 + q2
                bary[y, x, 0] = q0 / s
                bary[y, x, 1] = q1 / s


def rasteriser(tampons, ecran, prof, inv_w, id_depart=0):
    """Rasterise des triangles dans les tampons.

    ecran : (T, 3, 2) positions des sommets en pixels ; prof : (T, 3) z_ndc ;
    inv_w : (T, 3) 1 / w_clip ; id_depart : numéro donné au premier triangle dans le tampon.
    """
    if len(ecran):
        _rasteriser(np.ascontiguousarray(ecran, dtype=np.float64), np.ascontiguousarray(prof, dtype=np.float64),
                    np.ascontiguousarray(inv_w, dtype=np.float64), id_depart,
                    tampons.profondeur, tampons.triangle, tampons.bary)
