"""Génère par le code les modèles 3D des albums et les enregistre dans modeles/.

Pour chaque album : un .obj (sommets, normales, UV, triangles, un objet « o » par partie
animable), son .mtl, ses textures .png, et une entrée dans modeles/modeles.json
(pivots et axes des parties animées).

Repère commun à tous les modèles :
- unité = largeur de la pochette, origine au centre de la pochette ;
- la pochette occupe le carré [-0.5, 0.5] x [-0.5, 0.5] du plan y = 0 ;
- Y = normale sortante de la pochette (vers le haut), +X = droite de la pochette,
  +Z = bord bas de la pochette (côté spectateur) ; les objets « regardent » vers +Z.

Lancement : uv run python -m outils.generer_modeles
"""

import json
from functools import reduce
from pathlib import Path

import cv2
import numpy as np

from pochettes_ar.scene.maillage import Maillage, Materiau, Partie, calculer_normales, ecrire_obj

DOSSIER = Path(__file__).resolve().parent.parent / "modeles"


# --- Transformations 4x4 (pour placer les primitives) --------------------------------


def translation(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def echelle(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0])


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def compose(*matrices):
    """compose(A, B, C) = A @ B @ C : C est appliquée en premier."""
    return reduce(np.matmul, matrices)


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


# --- Textures (images 2D dessinées avec OpenCV, pas du rendu 3D) ----------------------


def texture_jante(chemin, taille=256):
    img = np.full((taille, taille, 3), 25, np.uint8)
    c = taille // 2
    cv2.circle(img, (c, c), int(c * 0.62), (200, 200, 200), -1, cv2.LINE_AA)
    for k in range(5):
        a = 2 * np.pi * k / 5
        bout = (int(c + c * 0.58 * np.cos(a)), int(c + c * 0.58 * np.sin(a)))
        cv2.line(img, (c, c), bout, (70, 70, 70), taille // 22, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(c * 0.16), (235, 235, 235), -1, cv2.LINE_AA)
    cv2.circle(img, (c, c), int(c * 0.62), (120, 120, 120), 3, cv2.LINE_AA)
    cv2.imwrite(str(chemin), img)


def texture_billet(chemin, largeur=600, hauteur=260):
    """Billet stylisé (volontairement pas une reproduction d'un vrai billet)."""
    vert_clair, vert, vert_fonce = (190, 225, 200), (95, 140, 85), (40, 75, 35)  # BGR
    img = np.full((hauteur, largeur, 3), vert_clair, np.uint8)
    for k, couleur in enumerate([vert_fonce, vert, vert_fonce]):
        m = 8 + 7 * k
        cv2.rectangle(img, (m, m), (largeur - m, hauteur - m), couleur, 3 if k != 1 else 1)
    for x in range(40, largeur - 40, 12):  # guillochis
        y = (hauteur // 2 + 70 * np.sin(np.linspace(0, 6 * np.pi, 60) + x / 30)).astype(int)
        pts = np.stack([np.linspace(x, x + 40, 60).astype(int), y], 1)
        cv2.polylines(img, [pts], False, (165, 205, 175), 1, cv2.LINE_AA)
    c = (largeur // 2, hauteur // 2)
    cv2.ellipse(img, c, (70, 88), 0, 0, 360, vert_clair, -1, cv2.LINE_AA)
    cv2.ellipse(img, c, (70, 88), 0, 0, 360, vert_fonce, 4, cv2.LINE_AA)
    cv2.putText(img, "$", (c[0] - 34, c[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 3.2, vert_fonce, 6, cv2.LINE_AA)
    for x, y in [(30, 62), (largeur - 62, 62), (30, hauteur - 30), (largeur - 62, hauteur - 30)]:
        cv2.putText(img, "1", (x + 8, y), cv2.FONT_HERSHEY_TRIPLEX, 1.4, vert_fonce, 3, cv2.LINE_AA)
    cv2.putText(img, "ONE DOLLAR", (largeur // 2 - 88, hauteur - 26), cv2.FONT_HERSHEY_TRIPLEX, 0.9, vert_fonce, 2, cv2.LINE_AA)
    cv2.putText(img, "NEVERMIND", (largeur // 2 - 80, 48), cv2.FONT_HERSHEY_TRIPLEX, 0.9, vert_fonce, 2, cv2.LINE_AA)
    cv2.imwrite(str(chemin), img)


def texture_visiere(chemin, largeur=512, hauteur=128):
    """Bandes de LED colorées ; l'animation fera défiler u."""
    img = np.zeros((hauteur, largeur, 3), np.uint8)
    teintes = (np.arange(largeur) * 180 // largeur).astype(np.uint8)
    hsv = np.stack([np.tile(teintes, (hauteur, 1)), np.full((hauteur, largeur), 255, np.uint8),
                    np.full((hauteur, largeur), 255, np.uint8)], -1)
    arc = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for y in range(hauteur // 2 - 30, hauteur // 2 + 30, 12):  # rangées de LED
        for x in range(6, largeur, 12):
            cv2.circle(img, (x, y), 4, arc[y, x].tolist(), -1, cv2.LINE_AA)
    cv2.imwrite(str(chemin), img)


def texture_ecran(chemin, largeur=512, hauteur=384):
    """Terminal vert ; l'animation fera défiler v (texture répétable verticalement)."""
    img = np.full((hauteur, largeur, 3), (20, 30, 10), np.uint8)
    rng = np.random.default_rng(1997)
    lignes = ["OK COMPUTER", "> fitter happier", "> more productive", "> no alarms", "> no surprises",
              "> calm", "C:\\>run paranoid.exe", "> loading android..."]
    for k, y in enumerate(range(28, hauteur, 24)):
        texte = lignes[k] if k < len(lignes) else "".join(rng.choice(list("01#=>_-*"), size=rng.integers(8, 30)))
        cv2.putText(img, texte, (16, y), cv2.FONT_HERSHEY_PLAIN, 1.4, (90, 255, 120), 1, cv2.LINE_AA)
    img[::3] = (img[::3] * 0.55).astype(np.uint8)  # lignes de balayage
    cv2.imwrite(str(chemin), img)


# --- Modèles --------------------------------------------------------------------------


def coccinelle():
    """Abbey Road : Coccinelle blanche ; avant vers +X, elle roule le long de X."""
    carrosserie = Materiau("carrosserie_blanche", (0.92, 0.92, 0.88), (0.6, 0.6, 0.6), 60)
    vitre = Materiau("vitre", (0.12, 0.15, 0.2), (0.9, 0.9, 0.9), 120)
    pneu = Materiau("pneu", (0.05, 0.05, 0.05), (0.1, 0.1, 0.1), 8)
    jante = Materiau("jante", (1, 1, 1), (0.5, 0.5, 0.5), 40, "jante.png")
    phare = Materiau("phare", (1.0, 0.95, 0.7), (1, 1, 1), 100)

    # demi-ellipsoïde : demi-longueur 0.16, hauteur 0.12, demi-largeur 0.085, posé à y = 0.035
    forme = compose(translation(0, 0.035, 0), echelle(0.16, 0.12, 0.085))
    nu = 48
    caisse = Maillage.fusion([sphere((0, 38), nu=nu, nv=10), sphere((62, 90), nu=nu, nv=8), disque(nu, haut=False)])
    parties = [
        Partie("carrosserie", transformer(caisse, forme), carrosserie),
        Partie("vitres", transformer(sphere((38, 62), nu=nu, nv=6), forme), vitre),
    ]
    phares = [transformer(sphere(nu=12, nv=6), compose(forme, translation(0.85, 0.25, s * 0.42), echelle(0.08, 0.1, 0.1)))
              for s in (-1, 1)]
    parties.append(Partie("phares", Maillage.fusion(phares), phare))

    pivots = {}
    rayon, largeur = 0.035, 0.025
    for nom, x, z in [("roue_av_g", 0.10, -0.075), ("roue_av_d", 0.10, 0.075),
                      ("roue_ar_g", -0.10, -0.075), ("roue_ar_d", -0.10, 0.075)]:
        # cylindre d'axe Y -> axe Z, centré sur le moyeu
        placement = compose(translation(x, rayon, z), rotation_x(np.pi / 2), translation(0, -largeur / 2, 0),
                            echelle(rayon, 1, rayon), echelle(1, largeur, 1))
        cote = transformer(cylindre(24, bouchons=False), placement)
        flancs = Maillage.fusion([transformer(disque(24, haut=False), placement),
                                  transformer(disque(24), compose(placement, translation(0, 1, 0)))])
        parties += [Partie(nom, cote, pneu), Partie(f"{nom}_jante", flancs, jante)]
        pivots[nom] = pivots[f"{nom}_jante"] = {"centre": [x, rayon, z], "axe": [0, 0, 1]}

    textures = {"jante.png": texture_jante}
    animation = ("La voiture avance le long de +X (traversée de la pochette) ; chaque roue et sa jante "
                 "tournent autour de leur pivot à la vitesse v / rayon.")
    return parties, textures, pivots, animation


def billet_hamecon():
    """Nevermind : billet suspendu à un hameçon, au-dessus de la pochette."""
    billet = Materiau("billet", (1, 1, 1), (0.05, 0.05, 0.05), 5, "billet.png")
    metal = Materiau("metal", (0.55, 0.55, 0.6), (0.9, 0.9, 0.9), 90)
    nylon = Materiau("nylon", (0.85, 0.85, 0.85), (0.4, 0.4, 0.4), 30)

    larg, haut, zb, yb = 0.30, 0.13, 0.015, 0.20  # billet vertical dans le plan z = zb
    e = 0.0005  # demi-épaisseur : deux faces dos à dos pour qu'il soit visible des deux côtés
    face = quad((-larg / 2, yb - haut / 2, zb + e), (larg, 0, 0), (0, haut, 0), 30, 12)
    dos = quad((larg / 2, yb - haut / 2, zb - e), (-larg, 0, 0), (0, haut, 0), 30, 12)

    r = 0.015  # hameçon : hampe verticale, courbure qui traverse le billet, pointe
    t = np.linspace(0, np.pi, 16)
    courbe = np.stack([np.zeros_like(t), 0.25 - r * np.sin(t), r - r * np.cos(t)], 1)
    trace = np.concatenate([[[0, 0.33, 0], [0, 0.29, 0]], courbe, [[0, 0.262, 2 * r], [0, 0.272, 2 * r]]])
    parties = [
        Partie("billet", Maillage.fusion([face, dos]), billet),
        Partie("hamecon", tube(trace, 0.003, 10), metal),
        Partie("fil", tube(np.linspace([0, 0.33, 0], [0, 0.65, 0], 8), 0.0012, 6), nylon),
    ]
    pivots = {"billet": {"centre": [0, 0.65, 0], "axe": [0, 0, 1]},
              "hamecon": {"centre": [0, 0.65, 0], "axe": [0, 0, 1]},
              "fil": {"centre": [0, 0.65, 0], "axe": [0, 0, 1]}}
    animation = ("Les sommets du billet ondulent (déplacement en z sinusoïdal selon x et le temps, normales "
                 "recalculées) ; l'ensemble se balance doucement autour du haut du fil.")
    return parties, {"billet.png": texture_billet}, pivots, animation


def pendules():
    """Rumours : deux boules suspendues à une potence, l'une mate, l'autre brillante."""
    bois = Materiau("bois", (0.45, 0.3, 0.18), (0.1, 0.1, 0.1), 10)
    mate = Materiau("boule_mate", (0.6, 0.12, 0.08), (0.05, 0.05, 0.05), 5)
    brillante = Materiau("boule_brillante", (0.92, 0.88, 0.78), (0.9, 0.9, 0.9), 120)
    ficelle = Materiau("ficelle", (0.15, 0.12, 0.1), (0, 0, 0), 1)

    h, rayon, yc = 0.56, 0.05, 0.25
    potence = Maillage.fusion([boite(0.02, h, 0.02, (-0.16, h / 2, 0)), boite(0.02, h, 0.02, (0.16, h / 2, 0)),
                               boite(0.36, 0.02, 0.02, (0, h, 0))])
    parties = [Partie("potence", potence, bois)]
    pivots = {}
    for nom, materiau, x in [("boule_mate", mate, -0.055), ("boule_brillante", brillante, 0.055)]:
        parties.append(Partie(nom, transformer(sphere(nu=40, nv=20), compose(translation(x, yc, 0), echelle(rayon, rayon, rayon))), materiau))
        fil = f"ficelle_{nom.split('_')[1]}"
        parties.append(Partie(fil, tube(np.linspace([x, h - 0.01, 0], [x, yc + rayon * 0.95, 0], 6), 0.0015, 6), ficelle))
        pivots[nom] = pivots[fil] = {"centre": [x, h - 0.01, 0], "axe": [0, 0, 1]}
    animation = ("Chaque boule et sa ficelle oscillent autour de leur pivot (axe Z, amplitude ~25°), "
                 "avec un déphasage entre les deux pendules.")
    return parties, {}, pivots, animation


def casque():
    """Random Access Memories : casque chromé à visière noire, regard vers +Z."""
    chrome = Materiau("chrome", (0.75, 0.75, 0.78), (1, 1, 1), 150)
    visiere = Materiau("visiere", (1, 1, 1), (1, 1, 1), 200, "visiere.png")
    noir = Materiau("col", (0.04, 0.04, 0.04), (0.3, 0.3, 0.3), 20)

    forme = compose(translation(0, 0.21, 0), echelle(0.12, 0.15, 0.13))
    # 7.5° entre deux méridiens partout pour que les bords des morceaux coïncident
    coque = Maillage.fusion([sphere((22, 90), nu=48, nv=14), sphere((-90, -18), nu=48, nv=14),
                             sphere((-18, 22), (60, 300), nu=32, nv=8)])
    parties = [
        Partie("casque", transformer(coque, forme), chrome),
        Partie("visiere", transformer(sphere((-18, 22), (-60, 60), nu=16, nv=8), forme), visiere),
        Partie("col", transformer(cylindre(32), echelle(0.065, 0.09, 0.065)), noir),
    ]
    pivots = {name: {"centre": [0, 0.21, 0], "axe": [0, 1, 0]} for name in ("casque", "visiere", "col")}
    animation = ("Le casque tourne lentement autour de l'axe Y ; la coordonnée u de la visière défile "
                 "(u + vitesse x t, texture répétée) pour animer les LED.")
    return parties, {"visiere.png": texture_visiere}, pivots, animation


def ordinateur():
    """OK Computer : moniteur cathodique beige et clavier, à rendre en ombrage plat."""
    plastique = Materiau("plastique_beige", (0.82, 0.78, 0.68), (0.1, 0.1, 0.1), 8)
    touche = Materiau("touches", (0.7, 0.67, 0.6), (0.15, 0.15, 0.15), 10)
    ecran = Materiau("ecran", (1, 1, 1), (0.7, 0.7, 0.7), 80, "ecran.png")

    zc = -0.04  # moniteur : boîtier 0.26 x 0.22 x 0.22 sur un pied, face avant en z = zc + 0.11
    moniteur = Maillage.fusion([boite(0.12, 0.03, 0.12, (0, 0.015, zc)), boite(0.26, 0.22, 0.22, (0, 0.14, zc)),
                                boite(0.18, 0.16, 0.06, (0, 0.14, zc - 0.13))])
    # dalle bombée : quad subdivisé, déplacé en z, normales recalculées
    dalle = quad((-0.1, 0.07, 0), (0.2, 0, 0), (0, 0.15, 0), 16, 12)
    u, v = (dalle.sommets[:, 0] / 0.1), ((dalle.sommets[:, 1] - 0.145) / 0.075)
    dalle.sommets[:, 2] = zc + 0.111 + 0.012 * (1 - u**2) * (1 - v**2)
    dalle.normales = calculer_normales(dalle.sommets, dalle.triangles)

    touches = [boite(0.022, 0.012, 0.022, (-0.135 + 0.03 * i + 0.01 * (r % 2), 0.018, 0.13 + 0.028 * r))
               for r in range(3) for i in range(9)]
    touches.append(boite(0.15, 0.012, 0.022, (0, 0.018, 0.13 + 0.028 * 3)))  # barre d'espace
    parties = [
        Partie("moniteur", moniteur, plastique),
        Partie("ecran", dalle, ecran),
        Partie("clavier", boite(0.30, 0.012, 0.13, (0, 0.006, 0.17)), plastique),
        Partie("touches", Maillage.fusion(touches), touche),
    ]
    pivots = {name: {"centre": [0, 0.0, 0.05], "axe": [0, 1, 0]} for name in ("moniteur", "ecran", "clavier", "touches")}
    animation = ("L'ensemble flotte (translation verticale sinusoïdale) et pivote doucement autour de Y ; "
                 "la coordonnée v de l'écran défile pour faire monter le texte. Ombrage plat conseillé.")
    return parties, {"ecran.png": texture_ecran}, pivots, animation


MODELES = {
    "the_beatles_abbey_road": ("coccinelle", coccinelle),
    "nirvana_nevermind": ("billet_hamecon", billet_hamecon),
    "fleetwood_mac_rumours": ("pendules", pendules),
    "daft_punk_random_access_memories": ("casque", casque),
    "radiohead_ok_computer": ("ordinateur", ordinateur),
}


def main():
    DOSSIER.mkdir(exist_ok=True)
    catalogue = {
        "_repere": "unité = largeur de la pochette ; pochette = carré [-0.5, 0.5]² du plan y = 0 ; "
                   "Y = normale sortante ; +X = droite ; +Z = bord bas (côté spectateur)",
    }
    for album, (nom, fabrique) in MODELES.items():
        parties, textures, pivots, animation = fabrique()
        for fichier, dessiner in textures.items():
            dessiner(DOSSIER / fichier)
        ecrire_obj(DOSSIER / f"{nom}.obj", parties)
        nb = sum(len(p.maillage.triangles) for p in parties)
        catalogue[album] = {
            "modele": f"{nom}.obj",
            "parties": [p.nom for p in parties],
            "pivots": pivots,
            "animation": animation,
            # rétro low-poly pour l'ordinateur : une normale par face
            "ombrage": "plat" if nom == "ordinateur" else "phong",
        }
        print(f"{album:34s} -> {nom}.obj  ({len(parties)} parties, {nb} triangles)")
    (DOSSIER / "modeles.json").write_text(json.dumps(catalogue, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
