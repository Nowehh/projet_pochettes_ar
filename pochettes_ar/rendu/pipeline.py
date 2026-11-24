"""Pipeline de rendu complet (M&S ch. 8), écrit par nous : aucune bibliothèque 3D.

Pour chaque image :
    rendre(fond, [Appel(objet, vue), ...])
        pour chaque objet (« appel de dessin ») :
            1. traitement des sommets : modèle -> caméra -> clip (matrices de camera.py) ;
            2. assemblage des triangles, rejet (derrière la caméra, faces arrière, hors écran),
               division perspective et viewport ;
            3. rasterisation + z-buffer (rasterisation.py), dans des tampons communs à tous
               les objets : ils se cachent correctement les uns les autres ;
        puis, une seule fois :
            4. shading différé : pour chaque pixel couvert, interpolation des attributs du
               triangle visible (position, normale, UV), texture, éclairage Blinn-Phong ;
            5. composition par-dessus l'image webcam.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from pochettes_ar.rendu import camera
from pochettes_ar.rendu.compositing import composer
from pochettes_ar.rendu.rasterisation import rasteriser
from pochettes_ar.rendu.shading import blinn_phong, normaliser
from pochettes_ar.rendu.textures import echantillonner_bilineaire
from pochettes_ar.rendu.transformations import matrice_normales
from pochettes_ar.rendu.zbuffer import Tampons

# Plans de coupe, en largeurs de pochette (l'unité de la pose et des modèles).
PRES, LOIN = 0.05, 100.0

# Lumière directionnelle fixée à la pochette (comme une lampe au-dessus de la table) :
# direction VERS la lumière, dans le repère pochette (Y sort de la pochette).
DIRECTION_LUMIERE = normaliser(np.array([0.4, 1.0, 0.6]))


@dataclass
class Appel:
    """Un « appel de dessin » : un objet, la pose de sa pochette, et l'état animé de l'objet
    (voir scene/animations.py) ; sans animation, l'objet est rendu tel que dans le fichier OBJ.

    - matrices_parties : matrice modèle -> pochette par partie (identité par défaut) ;
    - decalages_uv : décalage (du, dv) des coordonnées de texture par partie (texture qui défile) ;
    - sommets, normales : sommets déformés (V, 3) dans le repère du modèle, à la place de ceux
      de l'objet (animation par sommet, comme le billet qui ondule)."""

    objet: object  # ObjetRendu
    vue: np.ndarray  # (4, 4) pochette -> caméra (Pose.matrice())
    matrices_parties: dict = field(default_factory=dict)  # nom de partie -> (4, 4)
    decalages_uv: dict = field(default_factory=dict)  # nom de partie -> (du, dv)
    sommets: np.ndarray | None = None
    normales: np.ndarray | None = None


@dataclass
class _Prepare:
    """Ce qu'on garde d'un appel entre la rasterisation et le shading."""

    appel: Appel
    id_depart: int
    triangles: np.ndarray  # (n, 3) triangles gardés (indices de sommets)
    materiaux: np.ndarray  # (n,) leur matériau
    pos_camera: np.ndarray  # (V, 3) sommets dans le repère caméra
    normales_camera: np.ndarray  # (V, 3)
    uv: np.ndarray  # (V, 2) coordonnées de texture (décalées si la texture défile)


class Pipeline:
    def __init__(self, largeur, hauteur, K):
        self.largeur, self.hauteur = largeur, hauteur
        self.projection = camera.matrice_projection(K, largeur, hauteur, PRES, LOIN)
        self.viewport = camera.matrice_viewport(largeur, hauteur)
        self.tampons = Tampons(largeur, hauteur)
        self.stats = {}
        # Premier appel : Numba compile (ou recharge du cache) le rasteriseur maintenant,
        # plutôt qu'au milieu de la première image.
        rasteriser(Tampons(4, 4), np.zeros((1, 3, 2)), np.zeros((1, 3)), np.ones((1, 3)))

    def rendre(self, fond, appels):
        """Renvoie fond (BGR uint8) avec les objets rendus par-dessus."""
        debut = time.perf_counter()
        self.tampons.effacer()
        prepares, total = [], 0
        for appel in appels:
            prepare = self._sommets_et_rasterisation(appel, total)
            total += len(prepare.triangles)
            prepares.append(prepare)
        image, nb_pixels = self._shading_et_composition(fond, prepares)
        self.stats = {"triangles": total, "pixels": nb_pixels, "ms": 1000 * (time.perf_counter() - debut)}
        return image

    # --- Étapes 1 à 3 -------------------------------------------------------------------

    def _sommets_et_rasterisation(self, appel, id_depart):
        objet = appel.objet
        sommets = objet.sommets if appel.sommets is None else appel.sommets
        normales = objet.normales if appel.normales is None else appel.normales

        # 1. Traitement des sommets. Matrice modèle-vue de chaque partie :
        #    MV = M_vue · M_modele (le sommet subit d'abord M_modele, puis la pose).
        modeles = np.stack([appel.matrices_parties.get(nom, np.eye(4)) for nom in objet.noms_parties])
        mv = appel.vue @ modeles  # (P, 4, 4)
        mv_sommet = mv[objet.partie_sommet]  # (V, 4, 4) la matrice de la partie de chaque sommet
        pos_camera = np.einsum("vij,vj->vi", mv_sommet[:, :3, :3], sommets) + mv_sommet[:, :3, 3]
        normales_camera = normaliser(np.einsum("vij,vj->vi", matrice_normales(mv)[objet.partie_sommet], normales))
        # Projection : coordonnées de clip (x, y, z, w) = P (X, Y, Z, 1), avec w = Z.
        clip = np.concatenate([pos_camera, np.ones((len(pos_camera), 1))], axis=1) @ self.projection.T

        # 2. Assemblage des triangles et rejets.
        tri = objet.triangles
        w = clip[:, 3]
        # (a) Triangles dont un sommet est derrière le plan proche : rejetés en entier.
        #     Simplification : un vrai moteur les découperait (clipping, M&S 8.1.6). Ici,
        #     l'objet est posé sur la pochette, bien devant la caméra : le cas est rare.
        devant = (w[tri] > PRES).all(axis=1)
        # Division perspective (clip -> NDC), puis viewport (NDC -> pixels).
        w_sur = np.where(w > PRES, w, 1.0)[:, None]
        ndc = clip[:, :3] / w_sur
        ecran = np.concatenate([ndc, np.ones((len(ndc), 1))], axis=1) @ self.viewport.T  # (x_px, y_px, z_ndc, 1)
        e = ecran[tri]  # (T, 3, 4)
        # (b) Élimination des faces arrière (back-face culling). Nos triangles tournent dans le
        #     sens trigonométrique vus de l'extérieur. Dans l'image (y vers le bas), un triangle
        #     vu de face a donc une aire signée négative. Positive : on voit sa face arrière,
        #     elle est forcément cachée par une face avant (objet fermé) : inutile de la dessiner.
        aire = (e[:, 1, 0] - e[:, 0, 0]) * (e[:, 2, 1] - e[:, 0, 1]) - (e[:, 2, 0] - e[:, 0, 0]) * (e[:, 1, 1] - e[:, 0, 1])
        face_avant = aire < 0
        # (c) Triangles entièrement hors de l'image.
        dans_image = ((e[:, :, 0].max(1) >= 0) & (e[:, :, 0].min(1) <= self.largeur - 1)
                      & (e[:, :, 1].max(1) >= 0) & (e[:, :, 1].min(1) <= self.hauteur - 1))
        garde = devant & face_avant & dans_image

        # 3. Rasterisation et z-buffer.
        rasteriser(self.tampons, e[garde][:, :, :2], e[garde][:, :, 2], 1.0 / w[tri[garde]], id_depart)
        # Coordonnées de texture : un décalage constant par partie fait glisser la texture sur
        # la surface (elle est répétée au-delà de [0, 1], voir textures.py).
        decalages = np.array([appel.decalages_uv.get(nom, (0.0, 0.0)) for nom in objet.noms_parties], dtype=np.float64)
        uv = objet.uv + decalages[objet.partie_sommet]
        return _Prepare(appel, id_depart, tri[garde], objet.materiau_triangle[garde], pos_camera, normales_camera, uv)

    # --- Étapes 4 et 5 ------------------------------------------------------------------

    def _shading_et_composition(self, fond, prepares):
        ys, xs = np.nonzero(self.tampons.triangle >= 0)  # pixels couverts par un objet
        ids = self.tampons.triangle[ys, xs]
        b = self.tampons.bary[ys, xs]
        bary = np.stack([b[:, 0], b[:, 1], 1.0 - b[:, 0] - b[:, 1]], axis=1)  # (n, 3)
        couleurs = np.zeros((len(ys), 3))

        for p in prepares:
            # Pixels dont le triangle visible appartient à cet appel.
            ici = (ids >= p.id_depart) & (ids < p.id_depart + len(p.triangles))
            if not ici.any():
                continue
            t = ids[ici] - p.id_depart
            tri = p.triangles[t]  # (n, 3) sommets du triangle visible en chaque pixel
            bc = bary[ici][:, :, None]  # (n, 3, 1)

            # 4a. Interpolation des attributs (barycentriques déjà corrigées de la perspective).
            position = (bc * p.pos_camera[tri]).sum(axis=1)
            uv = (bc * p.uv[tri]).sum(axis=1)
            if p.appel.objet.ombrage == "plat":
                # Ombrage plat : une seule normale par face, celle du plan du triangle.
                a, b_, c = (p.pos_camera[tri[:, k]] for k in range(3))
                normale = normaliser(np.cross(b_ - a, c - a))
            else:
                # Ombrage de Phong : normale interpolée puis renormalisée en chaque pixel.
                normale = normaliser((bc * p.normales_camera[tri]).sum(axis=1))

            # 4b. Matériau et texture.
            objet = p.appel.objet
            mat = p.materiaux[t]
            kd = objet.kd[mat].copy()
            if objet.couleurs is not None:
                # Couleur par sommet, interpolée comme les autres attributs (modèles IA).
                kd *= (bc * objet.couleurs[tri]).sum(axis=1)
            for m in np.unique(mat):
                if objet.textures[m] is not None:
                    sel = mat == m
                    kd[sel] *= echantillonner_bilineaire(objet.textures[m], uv[sel])

            # 4c. Éclairage, calculé dans le repère caméra : l'œil est à l'origine, donc la
            #     direction vers l'œil est -position. La lumière, fixée à la pochette, est
            #     tournée par la pose.
            vers_lumiere = p.appel.vue[:3, :3] @ DIRECTION_LUMIERE
            vers_oeil = normaliser(-position)
            couleurs[ici] = blinn_phong(normale, vers_lumiere, vers_oeil, kd, objet.ks[mat], objet.ns[mat])

        # 5. Composition sur l'image webcam.
        return composer(fond, ys, xs, couleurs), len(ys)
