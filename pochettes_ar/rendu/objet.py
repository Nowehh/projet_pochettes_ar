"""Objet prêt à rendre : toutes les parties d'un modèle OBJ réunies dans des tableaux uniques
(l'équivalent de nos « vertex buffers »), avec l'indice de partie de chaque sommet (pour
animer les parties séparément) et l'indice de matériau de chaque triangle."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pochettes_ar.rendu.textures import charger_texture
from pochettes_ar.scene.maillage import charger_obj


@dataclass
class ObjetRendu:
    sommets: np.ndarray  # (V, 3) dans le repère du modèle (= repère pochette au repos)
    normales: np.ndarray  # (V, 3)
    uv: np.ndarray  # (V, 2)
    triangles: np.ndarray  # (T, 3) indices de sommets
    partie_sommet: np.ndarray  # (V,) indice de la partie de chaque sommet
    noms_parties: list  # nom de chaque partie (les « o » du fichier OBJ)
    materiau_triangle: np.ndarray  # (T,) indice du matériau de chaque triangle
    kd: np.ndarray  # (M, 3) couleur diffuse de chaque matériau
    ks: np.ndarray  # (M, 3) couleur spéculaire
    ns: np.ndarray  # (M,) exposant de Phong
    textures: list  # (M,) texture float32 RGB, ou None
    ombrage: str = "phong"  # "phong" : normale interpolée ; "plat" : normale de la face
    couleurs: np.ndarray | None = None  # (V, 3) couleur par sommet (modèles IA), ou None

    @classmethod
    def depuis_obj(cls, chemin, ombrage="phong"):
        chemin = Path(chemin)
        parties = charger_obj(chemin)

        materiaux = {}  # nom -> indice
        for p in parties:
            materiaux.setdefault(p.materiau.nom, (len(materiaux), p.materiau))
        textures_chargees = {}

        def texture(m):
            if not m.texture:
                return None
            if m.texture not in textures_chargees:
                textures_chargees[m.texture] = charger_texture(chemin.parent / m.texture)
            return textures_chargees[m.texture]

        decalages = np.cumsum([0] + [len(p.maillage.sommets) for p in parties[:-1]])
        liste = [m for _, m in sorted(materiaux.values(), key=lambda im: im[0])]
        return cls(
            sommets=np.concatenate([p.maillage.sommets for p in parties]),
            normales=np.concatenate([p.maillage.normales for p in parties]),
            uv=np.concatenate([p.maillage.uv for p in parties]),
            triangles=np.concatenate([p.maillage.triangles + d for p, d in zip(parties, decalages)]),
            partie_sommet=np.concatenate([np.full(len(p.maillage.sommets), i) for i, p in enumerate(parties)]),
            noms_parties=[p.nom for p in parties],
            materiau_triangle=np.concatenate([np.full(len(p.maillage.triangles), materiaux[p.materiau.nom][0])
                                              for p in parties]),
            kd=np.array([m.kd for m in liste], dtype=np.float64),
            ks=np.array([m.ks for m in liste], dtype=np.float64),
            ns=np.array([m.ns for m in liste], dtype=np.float64),
            textures=[texture(m) for m in liste],
            ombrage=ombrage,
            couleurs=np.concatenate([p.maillage.couleurs for p in parties]),
        )
