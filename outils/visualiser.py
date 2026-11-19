"""Visualiseur de contrôle des modèles de modeles/, avec matplotlib.

Outil provisoire en attendant notre moteur de rendu (pochettes_ar/rendu/). Il relit les
.obj avec notre chargeur, colore chaque triangle avec la texture échantillonnée au centre
de ses UV (sinon Kd) et un éclairage de Lambert calculé à partir des normales du fichier,
puis pose le modèle sur l'image de sa pochette (plan y = 0).

Lancement :
    uv run python -m outils.visualiser                 # les 5 modèles
    uv run python -m outils.visualiser casque --normales
    uv run python -m outils.visualiser --sortie apercu.png
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from pochettes_ar.scene.maillage import charger_obj

RACINE = Path(__file__).resolve().parent.parent
LUMIERE = np.array([0.4, 1.0, 0.7]) / np.linalg.norm([0.4, 1.0, 0.7])
AMBIANT = 0.3


def vers_mpl(p):
    """Notre repère (Y en haut) -> repère matplotlib (Z en haut) : (x, y, z) -> (x, -z, y)."""
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def lire_image(chemin):
    img = cv2.imread(str(chemin), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0


def echantillonner(img, uv):
    """Plus proche voisin, texture répétée ; v = 0 en bas de l'image."""
    h, w = img.shape[:2]
    x = (np.mod(uv[:, 0], 1.0) * (w - 1)).round().astype(int)
    y = ((1 - np.mod(uv[:, 1], 1.0)) * (h - 1)).round().astype(int)
    return img[y, x]


def polygones_pochette(image_album, n=48):
    """La pochette en n x n petits carrés colorés, dans le plan y = 0."""
    img = lire_image(image_album)
    t = np.linspace(-0.5, 0.5, n + 1)
    x0, z0 = np.meshgrid(t[:-1], t[:-1], indexing="ij")
    x0, z0, d = x0.ravel(), z0.ravel(), 1.0 / n
    coins = np.stack([np.stack([x0, 0 * x0, z0], 1), np.stack([x0 + d, 0 * x0, z0], 1),
                      np.stack([x0 + d, 0 * x0, z0 + d], 1), np.stack([x0, 0 * x0, z0 + d], 1)], 1)
    if img is None:
        return coins, np.full((len(coins), 3), 0.5)
    # haut de l'image = z = -0.5 (bord haut de la pochette)
    uv = np.stack([x0 + d / 2 + 0.5, 1 - (z0 + d / 2 + 0.5)], 1)
    return coins, echantillonner(img, uv)


def polygones_modele(parties, dossier):
    polys, couleurs = [], []
    for p in parties:
        m = p.maillage
        tris = m.sommets[m.triangles]
        texture = lire_image(dossier / p.materiau.texture) if p.materiau.texture else None
        if texture is not None:
            base = echantillonner(texture, m.uv[m.triangles].mean(axis=1)) * p.materiau.kd
        else:
            base = np.tile(p.materiau.kd, (len(tris), 1))
        n = m.normales[m.triangles].mean(axis=1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        lambert = np.clip(n @ LUMIERE, 0, 1)[:, None]
        polys.append(tris)
        couleurs.append(np.clip(base * (AMBIANT + (1 - AMBIANT) * lambert), 0, 1))
    return np.concatenate(polys), np.concatenate(couleurs)


def subdiviser(tris, couleurs, taille_max=0.025):
    """Coupe en 4 les triangles trop grands : le tri par peintre de matplotlib (par triangle
    entier) se trompe moins quand tous les triangles ont une taille comparable."""
    while True:
        aretes = np.linalg.norm(tris - np.roll(tris, 1, axis=1), axis=2).max(axis=1)
        grands = aretes > taille_max
        if not grands.any():
            return tris, couleurs
        a, b, c = tris[grands, 0], tris[grands, 1], tris[grands, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        enfants = np.concatenate([np.stack(t, 1) for t in [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]])
        tris = np.concatenate([tris[~grands], enfants])
        couleurs = np.concatenate([couleurs[~grands], np.tile(couleurs[grands], (4, 1))])


def afficher(ax, entree, catalogue_albums, aretes, normales, serre):
    chemin = RACINE / "modeles" / entree["modele"]
    parties = charger_obj(chemin)
    polys, couleurs = polygones_modele(parties, chemin.parent)
    nb = len(polys)
    bas, haut = polys.reshape(-1, 3).min(0), polys.reshape(-1, 3).max(0)
    if not aretes:
        polys, couleurs = subdiviser(polys, couleurs)
    if serre:  # sans la pochette, cadré sur le modèle
        coins, couleurs_p = np.zeros((0, 4, 3)), np.zeros((0, 3))
    else:
        album = catalogue_albums.get(entree["album"])
        coins, couleurs_p = polygones_pochette(RACINE / "albums" / album["fichier"] if album else None)

    toutes_couleurs = np.concatenate([couleurs_p, couleurs])
    # Une seule collection pour que matplotlib trie tous les polygones ensemble (peintre).
    collection = Poly3DCollection(
        list(vers_mpl(coins)) + list(vers_mpl(polys)),
        facecolors=toutes_couleurs,
        edgecolors="k" if aretes else toutes_couleurs,
        linewidths=0.15 if aretes else 0.2,
    )
    ax.add_collection3d(collection)

    if normales:
        segments = []
        for p in parties:
            m = p.maillage
            pas = max(1, len(m.sommets) // 300)
            debut = m.sommets[::pas]
            segments += list(np.stack([debut, debut + 0.03 * m.normales[::pas]], 1))
        ax.add_collection3d(Line3DCollection(vers_mpl(np.array(segments)), colors="tab:red", linewidths=0.6))

    # Mêmes unités sur les trois axes. Cadre : la pochette en x, z et la hauteur du modèle en y,
    # ou (--serre) la boîte englobante du modèle.
    if serre:
        bas, haut = vers_mpl(bas), vers_mpl(haut)
        lo, hi = np.minimum(bas, haut), np.maximum(bas, haut)
        centre, demi = (lo + hi) / 2, 0.55 * (hi - lo).max()
        lo, hi = centre - demi, centre + demi
    else:
        lo, hi = np.array([-0.5, -0.5, 0]), np.array([0.5, 0.5, max(0.25, 1.05 * haut[1])])
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=1.15)
    ax.set_title(f"{entree['modele']}\n{len(parties)} parties, {nb} triangles", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("-z")
    ax.set_zlabel("y")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modeles", nargs="*", help="noms des modèles (ex. casque) ; défaut : tous")
    parser.add_argument("--normales", action="store_true", help="affiche les normales (rouge)")
    parser.add_argument("--aretes", action="store_true", help="affiche les arêtes des triangles")
    parser.add_argument("--serre", action="store_true", help="cadre serré sur le modèle, sans la pochette")
    parser.add_argument("--elev", type=float, default=22)
    parser.add_argument("--azim", type=float, default=-60)
    parser.add_argument("--sortie", help="enregistre l'image au lieu d'ouvrir une fenêtre")
    args = parser.parse_args()

    catalogue = json.loads((RACINE / "modeles" / "modeles.json").read_text())
    albums = json.loads((RACINE / "albums" / "albums.json").read_text())
    entrees = [dict(v, album=k) for k, v in catalogue.items() if not k.startswith("_")]
    if args.modeles:
        voulus = {m.removesuffix(".obj") for m in args.modeles}
        entrees = [e for e in entrees if e["modele"].removesuffix(".obj") in voulus]
        if not entrees:
            parser.error(f"modèle inconnu ; choix : {[e['modele'] for e in catalogue.values() if isinstance(e, dict)]}")

    if args.sortie:
        plt.switch_backend("Agg")
    fig = plt.figure(figsize=(4.2 * len(entrees), 4.6))
    for k, entree in enumerate(entrees):
        ax = fig.add_subplot(1, len(entrees), k + 1, projection="3d")
        ax.view_init(elev=args.elev, azim=args.azim)
        afficher(ax, entree, albums, args.aretes, args.normales, args.serre)
    fig.tight_layout()
    if args.sortie:
        fig.savefig(args.sortie, dpi=130)
        print(f"image enregistrée : {args.sortie}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
