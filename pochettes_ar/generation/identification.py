"""Identification de l'album : quel est son nom, même s'il n'est pas écrit sur la pochette ?

Deux sources, de la plus sûre à la moins sûre :

1. Le texte lu sur la pochette (artiste et titre) : confirmé par une recherche MusicBrainz,
   qui en donne l'orthographe officielle. Le texte est sur la pochette : pas besoin d'autre
   vérification.

2. L'album que le modèle de vision dit reconnaître (pochettes célèbres sans texte, comme
   Abbey Road). Un modèle de langage peut « halluciner » : inventer avec assurance un album
   qui n'est pas le bon. On vérifie donc en deux temps :
   a. MusicBrainz doit connaître cet album chez cet artiste ;
   b. la pochette officielle de cet album (Cover Art Archive) doit être la même image que
      celle vue par la webcam : mêmes points ORB, reliés par une homographie (RANSAC),
      exactement comme la reconnaissance (vision/reconnaissance.py). Deux pochettes
      différentes ne passent pas ce test, même si elles se ressemblent.
   Si l'un des deux échoue, on préfère ne pas donner de nom plutôt qu'un nom faux.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pochettes_ar.generation import musicbrainz

RATIO_LOWE = 0.75
MIN_INLIERS = 25  # appariements cohérents avec une même homographie
MIN_PROPORTION = 0.3  # part des appariements retenus qui respectent l'homographie


@dataclass
class Identite:
    artiste: str
    album: str
    mbid: str  # identifiant du release-group MusicBrainz
    methode: str  # « texte » ou « reconnaissance vérifiée par l'image »

    @property
    def titre(self):
        return f"{self.artiste} - {self.album}"


def meme_pochette(image_a, image_b, taille=500):
    """Vrai si les deux images montrent la même pochette (ORB + ratio test + homographie RANSAC).
    Renvoie (vrai/faux, nombre d'inliers)."""
    orb = cv2.ORB_create(nfeatures=1500)
    gris = [cv2.cvtColor(cv2.resize(i, (taille, taille), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            for i in (image_a, image_b)]
    (kps_a, desc_a), (kps_b, desc_b) = (orb.detectAndCompute(g, None) for g in gris)
    if desc_a is None or desc_b is None or len(kps_a) < 2 or len(kps_b) < 2:
        return False, 0
    paires = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    bons = [p[0] for p in paires if len(p) == 2 and p[0].distance < RATIO_LOWE * p[1].distance]
    if len(bons) < MIN_INLIERS:
        return False, len(bons)
    src = np.float32([kps_a[m.queryIdx].pt for m in bons])
    dst = np.float32([kps_b[m.trainIdx].pt for m in bons])
    H, masque = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = 0 if H is None else int(masque.sum())
    return inliers >= MIN_INLIERS and inliers >= MIN_PROPORTION * len(bons), inliers


def identifier(description, image_bgr, confirmer_=musicbrainz.confirmer,
               telecharger_=musicbrainz.pochette_officielle):
    """Identité de l'album (voir le haut du fichier), ou None si on ne peut pas l'établir sûrement."""
    # 1. Texte lu sur la pochette.
    officiel = confirmer_(description.artiste, description.album)
    if officiel:
        return Identite(*officiel, methode="texte")
    # 2. Album reconnu par l'IA : MusicBrainz, puis comparaison avec la pochette officielle.
    officiel = confirmer_(description.reconnu_artiste, description.reconnu_album)
    if not officiel:
        return None
    reference = telecharger_(officiel[2])
    if reference is None:
        return None
    identique, _ = meme_pochette(image_bgr, reference)
    return Identite(*officiel, methode="reconnaissance vérifiée par l'image") if identique else None
