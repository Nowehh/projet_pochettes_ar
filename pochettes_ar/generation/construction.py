"""Étape 3 : composition JSON -> maillage, construit par notre code (M&S ch. 6 et 12).

Chaque pièce est une primitive canonique (pavé, sphère, cylindre, cône) qui tient dans le
cube unité centré [-0.5, 0.5]³, placée par sa matrice modèle :
    M_piece = T(position) · Rz · Ry · Rx · S(taille)
(les matrices s'appliquent de droite à gauche : mise à l'échelle, rotations, puis translation).

L'objet entier est ensuite ramené dans le repère du projet, quelle que soit l'unité choisie
par l'IA : centré sur la pochette, posé dessus (y min = 0), plus grande dimension = TAILLE.
Les pièces qui tournent sur elles-mêmes (roues, disques…) deviennent des parties animées,
avec leur pivot (centre de la pièce) et leur axe, comme les modèles faits à la main.
"""

import numpy as np

from pochettes_ar.generation.composition import couleur_rgb
from pochettes_ar.rendu.objet import ObjetRendu
from pochettes_ar.rendu.transformations import compose, echelle, rotation, translation
from pochettes_ar.scene.maillage import Materiau, Partie
from pochettes_ar.scene.primitives import boite, cone, cylindre, sphere, transformer

TAILLE = 0.6  # plus grande dimension de l'objet, en largeurs de pochette

# Primitives canoniques, toutes ramenées dans le cube [-0.5, 0.5]³.
_PRIMITIVES = {
    "boite": lambda: boite(1, 1, 1),
    "sphere": lambda: transformer(sphere(nu=24, nv=12), echelle(0.5, 0.5, 0.5)),
    # cylindre et cône de primitives.py : rayon 1, de y = 0 à y = 1 -> centrés, diamètre 1
    "cylindre": lambda: transformer(cylindre(24), compose(echelle(0.5, 1, 0.5), translation(0, -0.5, 0))),
    "cone": lambda: transformer(cone(24), compose(echelle(0.5, 1, 0.5), translation(0, -0.5, 0))),
}
_AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def matrice_piece(piece, indice=0):
    rx, ry, rz = np.radians(piece["rotation"])
    R = compose(rotation((0, 0, 1), rz), rotation((0, 1, 0), ry), rotation((1, 0, 0), rx))
    # Z-fighting : l'IA aligne souvent deux pièces face contre face (même plan). Les deux faces
    # ont alors la même profondeur, et le z-buffer choisit au hasard des arrondis, pixel par
    # pixel : la surface scintille. Agrandir chaque pièce d'une quantité différente (0.4 % par
    # indice) départage les faces sans changer la forme visible.
    marge = 1.0 + 0.004 * indice
    return compose(translation(*piece["position"]), R, echelle(*(np.array(piece["taille"]) * marge))), R


def construire(composition, taille=TAILLE):
    """Composition validée -> (ObjetRendu, entrée d'animation façon modeles.json)."""
    parties, pivots_bruts = [], {}
    for i, piece in enumerate(composition["pieces"]):
        M, R = matrice_piece(piece, i)
        maillage = transformer(_PRIMITIVES[piece["forme"]](), M)
        b = piece["brillance"]
        materiau = Materiau(f"piece_{i}", kd=couleur_rgb(piece["couleur"]),
                            ks=(0.8 * b,) * 3, ns=8 + 150 * b)
        nom = f"piece_{i}"
        parties.append(Partie(nom, maillage, materiau))
        if piece["tourne_sur_soi"] in _AXES:
            # Un cylindre ou un cône qui tourne (roue, disque, hélice) tourne autour de son propre
            # axe, quel que soit l'axe écrit par l'IA ; une autre forme tourne autour de l'axe
            # du monde indiqué.
            if piece["forme"] in ("cylindre", "cone"):
                axe = R[:3, :3] @ np.array([0.0, 1.0, 0.0])
            else:
                axe = np.array(_AXES[piece["tourne_sur_soi"]], dtype=np.float64)
            pivots_bruts[nom] = (np.array(piece["position"], dtype=np.float64), axe)

    # Normalisation : N = S(k) · T(-centre) amène l'objet sur la pochette, à la bonne taille.
    tous = np.concatenate([p.maillage.sommets for p in parties])
    mini, maxi = tous.min(axis=0), tous.max(axis=0)
    k = taille / max(float((maxi - mini).max()), 1e-9)
    N = compose(echelle(k, k, k), translation(-(mini[0] + maxi[0]) / 2, -mini[1], -(mini[2] + maxi[2]) / 2))
    for p in parties:
        p.maillage = transformer(p.maillage, N)
    pivots = {nom: {"centre": (N @ [*c, 1.0])[:3].tolist(), "axe": (a / np.linalg.norm(a)).tolist()}
              for nom, (c, a) in pivots_bruts.items()}

    entree = {"modele": "compose.json", "animation": composition["animation"], "pivots": pivots,
              "ombrage": "phong"}
    return ObjetRendu.depuis_parties(parties), entree
