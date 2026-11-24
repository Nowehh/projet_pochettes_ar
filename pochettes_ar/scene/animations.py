"""Animations par album : temps t (secondes) -> état animé de l'objet (M&S ch. 6 et 16).

Une animation ne modifie jamais le fichier OBJ : à chaque image, elle calcule
    - une matrice modèle -> pochette par partie (animation « rigide » : chaque partie bouge
      d'un bloc, autour du pivot donné dans modeles/modeles.json) ;
    - éventuellement un décalage des coordonnées de texture par partie (texture qui défile) ;
    - éventuellement de nouveaux sommets et de nouvelles normales (déformation, comme le
      billet qui ondule).
Le résultat se passe tel quel à Appel(objet, vue, **etat) (voir rendu/pipeline.py).

Les matrices se composent de droite à gauche : compose(A, B) applique B puis A. Une partie
qui tourne sur elle-même puis est déplacée avec tout l'objet s'écrit donc
    compose(deplacement_de_l_objet, rotation_autour(pivot, axe, angle)).
"""

from pathlib import Path

import numpy as np

from pochettes_ar.rendu.transformations import compose, rotation_autour, translation


def animer(entree, objet, t):
    """État animé de l'objet décrit par `entree` (une entrée de modeles.json) à l'instant t.
    Renvoie un dict (matrices_parties, decalages_uv, sommets, normales) ; vide si le modèle
    n'a pas d'animation."""
    fonction = ANIMATIONS.get(Path(entree["modele"]).stem)
    return fonction(entree, objet, t) if fonction else {}


def _pivot(entree, partie):
    p = entree["pivots"][partie]
    return np.array(p["centre"], dtype=np.float64), np.array(p["axe"], dtype=np.float64)


# --- Abbey Road : la Coccinelle traverse la pochette ----------------------------------------

VITESSE_VOITURE = 0.15  # largeurs de pochette par seconde
RAYON_ROUE = 0.035
TRAJET = (-0.7, 0.7)  # elle entre par la gauche, sort par la droite, et recommence


def coccinelle(entree, objet, t):
    distance = VITESSE_VOITURE * t
    debut, fin = TRAJET
    x = debut + distance % (fin - debut)
    deplacement = translation(x, 0, 0)
    # Roulement sans glissement : une roue de rayon r qui avance de d a tourné de d / r radians.
    # La voiture va vers +X, roue d'axe +Z : le haut de la roue part vers l'avant, ce qui est
    # une rotation d'angle négatif autour de +Z (règle de la main droite).
    angle = -distance / RAYON_ROUE
    matrices = {}
    for nom in objet.noms_parties:
        if nom in entree["pivots"]:
            centre, axe = _pivot(entree, nom)
            matrices[nom] = compose(deplacement, rotation_autour(centre, axe, angle))
        else:
            matrices[nom] = deplacement
    return {"matrices_parties": matrices}


# --- Nevermind : le billet ondule au bout du fil -----------------------------------------------

DEMI_LARGEUR_BILLET = 0.15


def billet_hamecon(entree, objet, t):
    # 1. Déformation des sommets du billet, dans le repère du modèle : déplacement en z
    #        f(x, t) = A (x / l)² sin(k x - w t),
    #    une onde qui court le long du billet, nulle au centre (là où l'hameçon le tient)
    #    et de plus en plus ample vers les bords.
    A, k, w = 0.012, 25.0, 3.0
    sommets, normales = objet.sommets.copy(), objet.normales.copy()
    billet = objet.partie_sommet == objet.noms_parties.index("billet")
    x = sommets[billet, 0]
    s = x / DEMI_LARGEUR_BILLET
    phase = k * x - w * t
    sommets[billet, 2] += A * s**2 * np.sin(phase)
    # 2. Normales recalculées : la surface z = z0 + f(x, y) a pour normale (-df/dx, -df/dy, 1),
    #    à normaliser (ici f ne dépend pas de y). Le dos du billet garde son orientation (-z).
    df_dx = A * (2 * s / DEMI_LARGEUR_BILLET * np.sin(phase) + s**2 * k * np.cos(phase))
    signe = np.sign(objet.normales[billet, 2])
    n = np.stack([-df_dx, np.zeros_like(x), np.ones_like(x)], axis=1) * signe[:, None]
    normales[billet] = n / np.linalg.norm(n, axis=1, keepdims=True)

    # 3. L'ensemble (billet, hameçon, fil) se balance autour du haut du fil.
    centre, axe = _pivot(entree, "fil")
    balancement = rotation_autour(centre, axe, np.radians(6) * np.sin(1.3 * t))
    return {"matrices_parties": {nom: balancement for nom in objet.noms_parties},
            "sommets": sommets, "normales": normales}


# --- Rumours : deux pendules déphasés ----------------------------------------------------------


def pendules(entree, objet, t):
    # Petites oscillations d'un pendule simple : angle(t) = amplitude x sin(w t + phase),
    # avec w = sqrt(g / longueur) ; on prend w à l'œil plutôt que la vraie valeur.
    amplitude, w = np.radians(25), 2.2
    matrices = {}
    for nom in objet.noms_parties:
        if nom not in entree["pivots"]:
            continue  # la potence ne bouge pas
        phase = 0.0 if nom.endswith("mate") else np.pi / 2  # boule et ficelle d'un même pendule
        centre, axe = _pivot(entree, nom)
        matrices[nom] = rotation_autour(centre, axe, amplitude * np.sin(w * t + phase))
    return {"matrices_parties": matrices}


# --- Random Access Memories : le casque tourne, les LED défilent ---------------------------


def casque(entree, objet, t):
    centre, axe = _pivot(entree, "casque")
    tour = rotation_autour(centre, axe, 0.6 * t)  # 0.6 rad/s : un tour en ~10 s
    # La texture de la visière est répétée en u : décaler u fait glisser les LED sur la visière.
    return {"matrices_parties": {nom: tour for nom in objet.noms_parties},
            "decalages_uv": {"visiere": (0.25 * t, 0.0)}}


# --- OK Computer : l'ordinateur flotte, le texte monte à l'écran ---------------------------


def ordinateur(entree, objet, t):
    centre, axe = _pivot(entree, "moniteur")
    # Pivote d'un côté à l'autre autour de Y, puis flotte au-dessus de la pochette.
    pivote = rotation_autour(centre, axe, np.radians(20) * np.sin(0.7 * t))
    flotte = translation(0, 0.04 * (1 + np.sin(1.5 * t)), 0)
    M = compose(flotte, pivote)
    # Un pixel de l'écran affiche la texture en v + dv : quand dv diminue, le contenu monte.
    return {"matrices_parties": {nom: M for nom in objet.noms_parties},
            "decalages_uv": {"ecran": (0.0, -0.1 * t)}}


# --- Modèles générés par IA (cache/) : un seul bloc, sans pivot ------------------------------


def ia(entree, objet, t):
    # Rotation lente autour de l'axe Y passant par le centre de la pochette, puis flottement.
    tour = rotation_autour((0, 0, 0), (0, 1, 0), 0.5 * t)
    flotte = translation(0, 0.03 * (1 + np.sin(1.5 * t)), 0)
    M = compose(flotte, tour)
    return {"matrices_parties": {nom: M for nom in objet.noms_parties}}


ANIMATIONS = {
    "ia": ia,
    "coccinelle": coccinelle,
    "billet_hamecon": billet_hamecon,
    "pendules": pendules,
    "casque": casque,
    "ordinateur": ordinateur,
}
