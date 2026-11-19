"""Surimpression 2D de contrôle sur l'image webcam : contour et titre des pochettes détectées.

Ce n'est pas du rendu 3D : simple annotation de débogage de la vision, faite avec OpenCV.
"""

import cv2
import numpy as np

from pochettes_ar.vision.calibration import CHAMP_HORIZONTAL_DEFAUT
from pochettes_ar.vision.pose import projeter

VERT = (80, 220, 60)  # BGR
POLICE = cv2.FONT_HERSHEY_SIMPLEX  # polices Hershey : ASCII seulement


def dessiner_detection(image, detection, pose=None):
    coins = detection.coins.round().astype(np.int32)

    # Contour : les 4 coins projetés par l'homographie, reliés dans l'ordre.
    cv2.polylines(image, [coins], isClosed=True, color=VERT, thickness=3, lineType=cv2.LINE_AA)
    # Le coin haut-gauche de la pochette est marqué d'un point : on voit ainsi son orientation.
    cv2.circle(image, tuple(coins[0]), 7, VERT, -1, cv2.LINE_AA)

    # Titre au-dessus du point le plus haut du contour, sur un bandeau lisible.
    details = f"{detection.inliers} inliers / {detection.votes} votes"
    if pose is not None:
        # distance en largeurs de pochette ; erreur de reprojection des 4 coins en pixels
        details += f"   d = {np.linalg.norm(pose.tvec):.1f} L   reproj {pose.erreur:.1f} px"
    lignes = [detection.titre, details]
    x, y = coins[np.argmin(coins[:, 1])]
    y -= 12
    if y < 110:  # trop près du haut (bandeau de statistiques) : titre sous la pochette
        x, y = coins[np.argmax(coins[:, 1])]
        y += 62
    for texte, echelle in reversed(list(zip(lignes, (0.8, 0.5)))):
        (l, h), base = cv2.getTextSize(texte, POLICE, echelle, 2)
        x0 = int(np.clip(x - l // 2, 0, max(0, image.shape[1] - l)))
        y0 = int(max(y, h + 4))
        cv2.rectangle(image, (x0 - 4, y0 - h - 4), (x0 + l + 4, y0 + base), VERT, -1)
        cv2.putText(image, texte, (x0, y0), POLICE, echelle, (0, 0, 0), 2 if echelle > 0.6 else 1, cv2.LINE_AA)
        y = y0 - h - base - 8


# Axes du repère de la pochette : X rouge, Y vert, Z bleu (couleurs en BGR).
AXES = [("X", (1, 0, 0), (40, 40, 230)), ("Y", (0, 1, 0), (60, 200, 60)), ("Z", (0, 0, 1), (230, 120, 30))]


def dessiner_axes(image, pose, K, longueur=0.5):
    """Projette l'origine et le bout des 3 axes (longueur en largeurs de pochette) et les trace.

    Vérifications attendues si la pose est juste : X longe le bord haut vers la droite, Z
    descend vers le bord bas, Y sort perpendiculairement de la pochette vers la caméra.
    """
    points = np.array([[0, 0, 0]] + [axe for _, axe, _ in AXES], dtype=np.float64) * longueur
    pixels, z = projeter(points, pose, K)
    if (z <= 1e-6).any():
        return  # un point derrière la caméra : la projection n'a pas de sens
    origine = tuple(pixels[0].round().astype(int))
    for (nom, _, couleur), bout in zip(AXES, pixels[1:]):
        bout = tuple(bout.round().astype(int))
        cv2.arrowedLine(image, origine, bout, (0, 0, 0), 7, cv2.LINE_AA, tipLength=0.12)
        cv2.arrowedLine(image, origine, bout, couleur, 4, cv2.LINE_AA, tipLength=0.12)
        cv2.putText(image, nom, (bout[0] + 6, bout[1] - 6), POLICE, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, nom, (bout[0] + 6, bout[1] - 6), POLICE, 0.9, couleur, 2, cv2.LINE_AA)


def dessiner_infos(image, stats, ips, calibration=None, rendu=None):
    """Bandeau en haut : points ORB, appariements après ratio test, 3 meilleurs votes, images/s, K."""
    meilleurs = sorted(stats.get("votes", {}).items(), key=lambda kv: -kv[1])[:3]
    votes = "   ".join(f"{titre.split(' - ', 1)[-1]} : {v}" for titre, v in meilleurs)
    lignes = [f"{ips:4.1f} img/s   points ORB : {stats.get('points', 0)}   "
              f"apres ratio test : {stats.get('appariements', 0)}",
              f"votes : {votes or '-'}"]
    if calibration is not None:
        K = calibration.K
        source = f"approximee (champ {CHAMP_HORIZONTAL_DEFAUT:.0f} deg)" if calibration.approximee else "calibree"
        lignes.append(f"K {source} : f = {K[0, 0]:.0f} px, centre = ({K[0, 2]:.0f}, {K[1, 2]:.0f})")
    if rendu:
        lignes.append(f"rendu : {rendu['triangles']} triangles, {rendu['pixels']} pixels, {rendu['ms']:.1f} ms")
    largeur = max(cv2.getTextSize(t, POLICE, 0.55, 1)[0][0] for t in lignes) + 20
    zone = image[:12 + 24 * len(lignes), :largeur]
    zone[:] = (zone * 0.35).astype(image.dtype)  # fond assombri, le texte reste lisible
    for k, texte in enumerate(lignes):
        cv2.putText(image, texte, (10, 22 + 24 * k), POLICE, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
