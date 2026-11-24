"""Reconnaissance des pochettes d'album dans une image webcam.

Le pipeline a deux phases.

Indexation (une seule fois, au démarrage), pour chaque pochette de référence de albums/ :
    1. on détecte des points d'intérêt ORB et on calcule leurs descripteurs ;
    2. on range tous les descripteurs dans une seule « base », en notant pour chacun
       à quelle pochette il appartient.

Reconnaissance (à chaque image webcam) :
    1. prétraitement : niveaux de gris et réduction de taille ;
    2. points d'intérêt ORB et descripteurs de l'image ;
    3. appariement avec la base : pour chaque descripteur de l'image, on cherche ses deux
       plus proches voisins et on applique le ratio test de Lowe ;
    4. vote : chaque appariement retenu vote pour la pochette de son voisin ; les pochettes
       les plus votées deviennent les candidates ;
    5. vérification géométrique : pour chaque candidate, on estime par RANSAC une homographie
       référence -> image. Si assez d'appariements la respectent et si la pochette projetée
       a une forme plausible, la détection est acceptée.

On obtient l'identité de la pochette, ses 4 coins dans l'image et l'homographie. Les coins
serviront ensuite à solvePnP pour calculer la pose 3D (voir pose.py).

OpenCV est autorisé ici : c'est la partie vision du projet, pas le rendu.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# --- Paramètres -----------------------------------------------------------------------
# Les valeurs ont été choisies pour une webcam en 720p ; elles se règlent ici.

# Côté (en pixels) auquel on réduit les images de référence. Les fichiers font 1200 px,
# bien plus que la taille d'une pochette tenue devant la webcam : réduire accélère le
# calcul et rapproche l'échelle de celle observée (ORB gère ensuite le reste par sa pyramide).
TAILLE_REFERENCE = 600

# Largeur maximale de l'image webcam analysée. Au-delà, on réduit. Le résultat (coins,
# homographie) est ensuite ramené à la taille d'origine. Réduire une image 720p à 800 px
# fait gagner ~4 ms, mais les pochettes lointaines (< 200 px) ne sont alors plus reconnues
# (essais sur scènes synthétiques : 45/60 à 800 px, 60/60 à 1280 px).
LARGEUR_ANALYSE = 1280

# Nombre maximal de points ORB par pochette de référence, puis par image webcam.
NB_POINTS_REFERENCE = 1500
NB_POINTS_IMAGE = 2000

# Niveaux de la pyramide ORB des références (facteur 1.2 entre deux niveaux). Avec 12
# niveaux, la référence de 600 px est aussi décrite jusqu'à 600 / 1.2^11 ≈ 80 px : on
# reconnaît des pochettes petites dans l'image. La pyramide de l'image garde 8 niveaux.
NIVEAUX_REFERENCE = 12

# Ratio test de Lowe : on garde un appariement si
#   distance(1er voisin) < RATIO_LOWE x distance(2e voisin).
# Lowe propose 0.8 pour SIFT ; 0.75 est un choix courant pour ORB.
RATIO_LOWE = 0.75

# Vote : nombre minimal de voix pour qu'une pochette soit candidate, et nombre maximal
# de candidates que l'on vérifie géométriquement (la vérification RANSAC est la plus coûteuse).
MIN_VOTES = 10
NB_CANDIDATES = 3

# RANSAC : un appariement est « inlier » si l'homographie l'envoie à moins de
# SEUIL_RANSAC pixels de l'endroit où on l'a observé (dans l'image analysée).
SEUIL_RANSAC = 5.0
# Acceptation : au moins MIN_INLIERS inliers et au moins MIN_PROPORTION_INLIERS
# des appariements votants.
MIN_INLIERS = 15
MIN_PROPORTION_INLIERS = 0.25

# Une pochette projetée plus petite que cette aire (en pixels de l'image analysée) est rejetée.
AIRE_MIN = 40 * 40


@dataclass
class Reference:
    """Une pochette indexée."""

    id: str  # clé de albums.json, ex. "nirvana_nevermind"
    titre: str  # "Nirvana - Nevermind"
    largeur: int  # taille de l'image de référence réduite, en pixels
    hauteur: int
    nb_points: int


@dataclass
class Detection:
    """Une pochette trouvée dans l'image webcam."""

    id: str
    titre: str
    # Coins de la pochette dans l'image webcam, en pixels, dans l'ordre :
    # haut-gauche, haut-droit, bas-droit, bas-gauche (ordre de l'image de référence).
    coins: np.ndarray  # (4, 2) float32
    # Homographie 3x3 : pixels de la référence (TAILLE_REFERENCE) -> pixels de l'image webcam.
    homographie: np.ndarray
    votes: int  # appariements ayant voté pour cette pochette
    inliers: int  # appariements cohérents avec l'homographie
    aspect: float = 1.0  # hauteur / largeur de la pochette (1 pour une pochette carrée)


class Reconnaisseur:
    def __init__(self, dossier_albums="albums"):
        dossier_albums = Path(dossier_albums)
        table = json.loads((dossier_albums / "albums.json").read_text())

        # ORB = « Oriented FAST and Rotated BRIEF » (Rublee et al., 2011).
        #  - FAST détecte des coins : un pixel dont un arc de pixels du cercle qui l'entoure
        #    est nettement plus clair ou plus sombre que lui.
        #  - Les coins sont classés par le score de Harris et cherchés sur une pyramide
        #    d'images (versions de plus en plus réduites, facteur 1.2) : on retrouve donc
        #    la pochette à différentes distances (invariance à l'échelle).
        #  - Chaque point reçoit une orientation (direction du centroïde d'intensité de son
        #    voisinage). Le descripteur est calculé dans ce repère tourné : il est invariant
        #    à la rotation.
        #  - Le descripteur BRIEF compte 256 bits (32 octets). Chaque bit compare l'intensité
        #    de deux pixels du voisinage lissé. On compare deux descripteurs avec la distance
        #    de Hamming (nombre de bits différents), ce qui est très rapide.
        self.orb_reference = cv2.ORB_create(nfeatures=NB_POINTS_REFERENCE, nlevels=NIVEAUX_REFERENCE)
        self.orb_image = cv2.ORB_create(nfeatures=NB_POINTS_IMAGE)

        # Appariement par force brute avec la distance de Hamming : chaque descripteur de
        # l'image est comparé à tous ceux de la base. Avec ~5 x 1500 descripteurs, c'est
        # assez rapide ; pour des centaines d'albums, on passerait à un index LSH (FLANN).
        self.apparieur = cv2.BFMatcher(cv2.NORM_HAMMING)

        # --- Indexation ---
        # La « base » : tous les descripteurs de toutes les pochettes, empilés, avec pour
        # chacun sa position dans la référence et l'indice de sa pochette.
        self.references = []
        self.base_points = np.zeros((0, 2), np.float32)  # (N, 2) position dans la référence
        self.base_descripteurs = np.zeros((0, 32), np.uint8)  # (N, 32)
        self.base_proprietaires = np.zeros(0, np.int64)  # (N,) indice dans self.references
        for id_album, infos in table.items():
            image = cv2.imread(str(dossier_albums / infos["fichier"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(dossier_albums / infos["fichier"])
            self.ajouter_reference(id_album, f"{infos['artiste']} - {infos['album']}", image)

        # Statistiques de la dernière image, pour l'affichage et l'oral.
        self.stats = {}

    def ajouter_reference(self, id_album, titre, image):
        """Indexe une pochette (image en niveaux de gris ou BGR). Sert au démarrage pour albums/,
        et en cours de route pour une pochette inconnue découverte par la webcam : son image
        redressée devient sa référence. Renvoie False si elle n'a pas assez de texture."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        facteur = TAILLE_REFERENCE / max(image.shape)
        image = cv2.resize(image, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_AREA)

        kps, desc = self.orb_reference.detectAndCompute(image, None)
        if desc is None or len(kps) < MIN_INLIERS:
            return False  # pochette sans texture (ex. unie) : impossible à reconnaître avec ORB
        hauteur, largeur = image.shape
        self.references.append(Reference(id_album, titre, largeur, hauteur, len(kps)))
        self.base_points = np.concatenate([self.base_points, np.float32([kp.pt for kp in kps])])
        self.base_descripteurs = np.concatenate([self.base_descripteurs, desc])
        self.base_proprietaires = np.concatenate([self.base_proprietaires,
                                                  np.full(len(kps), len(self.references) - 1)])
        return True

    def renommer(self, id_album, titre):
        for ref in self.references:
            if ref.id == id_album:
                ref.titre = titre

    def reconnaitre(self, image_bgr):
        """Renvoie la liste des pochettes détectées, triées par nombre d'inliers décroissant."""

        # --- Étape 1 : prétraitement ---
        # ORB travaille sur l'intensité seule. Réduire l'image accélère détection et
        # appariement ; on gardera le facteur pour revenir aux pixels d'origine.
        gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        facteur = min(1.0, LARGEUR_ANALYSE / gris.shape[1])
        if facteur < 1.0:
            gris = cv2.resize(gris, None, fx=facteur, fy=facteur, interpolation=cv2.INTER_AREA)

        # --- Étape 2 : points d'intérêt et descripteurs de l'image ---
        kps, desc = self.orb_image.detectAndCompute(gris, None)
        self.stats = {"points": len(kps), "appariements": 0, "votes": {}}
        if desc is None or len(kps) < MIN_INLIERS:
            return []
        points_image = np.float32([kp.pt for kp in kps])

        # --- Étape 3 : appariement + ratio test de Lowe ---
        # Pour chaque descripteur de l'image, on récupère ses 2 plus proches voisins dans la
        # base (k = 2). Un appariement est fiable si le 1er voisin est nettement meilleur que
        # le 2e. S'ils sont presque à égalité, le point est ambigu (motif répété, zone sans
        # texture, point qui n'est sur aucune pochette) : on le rejette.
        # La base contient toutes les pochettes : un point qui ressemble autant à deux
        # pochettes différentes est donc aussi éliminé, ce qui rend le vote plus discriminant.
        if len(self.base_descripteurs) < 2:
            return []  # base vide : aucune pochette indexée
        paires = self.apparieur.knnMatch(desc, self.base_descripteurs, k=2)
        bons = [p[0] for p in paires if len(p) == 2 and p[0].distance < RATIO_LOWE * p[1].distance]
        self.stats["appariements"] = len(bons)
        if len(bons) < MIN_VOTES:
            return []
        indices_image = np.array([m.queryIdx for m in bons])  # indice du point dans l'image
        indices_base = np.array([m.trainIdx for m in bons])  # indice du point dans la base

        # --- Étape 4 : vote ---
        # Chaque bon appariement vote pour la pochette à laquelle appartient son voisin.
        # Le vote est rapide mais ne regarde pas la géométrie : 30 voix réparties au hasard
        # dans l'image ne prouvent rien. Il sert seulement à choisir quelles pochettes
        # vérifier à l'étape 5, qui est plus coûteuse.
        pochette_votee = self.base_proprietaires[indices_base]
        votes = np.bincount(pochette_votee, minlength=len(self.references))
        self.stats["votes"] = {self.references[i].titre: int(v) for i, v in enumerate(votes) if v}
        candidates = [i for i in np.argsort(-votes)[:NB_CANDIDATES] if votes[i] >= MIN_VOTES]

        # Matrice de mise à l'échelle : pixels de l'image analysée -> pixels d'origine.
        retour_echelle = np.diag([1 / facteur, 1 / facteur, 1.0])

        detections = []
        for i in candidates:
            ref = self.references[i]
            selection = pochette_votee == i
            # Correspondances (point de la référence) <-> (point de l'image) pour cette pochette.
            src = self.base_points[indices_base[selection]]
            dst = points_image[indices_image[selection]]

            # --- Étape 5 : vérification géométrique par RANSAC ---
            # La pochette est plane. L'image d'un plan vu par une caméra (sténopé) est donc
            # liée à la référence par une homographie H (matrice 3x3, 8 degrés de liberté,
            # définie à un facteur près) : en coordonnées homogènes, p_image ~ H p_reference.
            #
            # Quatre correspondances suffisent pour calculer H, mais une partie des
            # appariements est fausse (outliers). RANSAC (Fischler & Bolles, 1981) :
            #   a. tirer 4 correspondances au hasard et calculer l'homographie qui les relie ;
            #   b. compter les correspondances qu'elle explique à SEUIL_RANSAC px près (inliers) ;
            #   c. recommencer et garder l'homographie qui a le plus d'inliers ;
            #   d. la réestimer par moindres carrés sur tous ses inliers.
            # Le nombre de tirages s'adapte à la proportion d'inliers observée, pour avoir
            # une probabilité « confidence » de faire au moins un tirage sans outlier.
            H, masque = cv2.findHomography(src, dst, cv2.RANSAC, SEUIL_RANSAC,
                                           maxIters=2000, confidence=0.995)
            if H is None:
                continue
            inliers = int(masque.sum())

            # Critère 1 : assez d'inliers, en nombre absolu et en proportion des votes.
            # De faux appariements dispersés n'ont presque aucune chance de tous respecter
            # une même homographie : beaucoup d'inliers prouvent donc la présence de la pochette.
            if inliers < MIN_INLIERS or inliers < MIN_PROPORTION_INLIERS * len(src):
                continue

            # On projette les 4 coins de la référence dans l'image : x' = H x, puis on
            # divise par la 3e coordonnée homogène (perspectiveTransform fait les deux).
            coins_ref = np.float32([[0, 0], [ref.largeur, 0], [ref.largeur, ref.hauteur], [0, ref.hauteur]])
            coins = cv2.perspectiveTransform(coins_ref[None], H)[0]

            # Critère 2 : la pochette projetée doit avoir une forme physiquement possible.
            # Un carré vu en perspective reste un quadrilatère convexe, non retourné (pas de
            # miroir : la pochette est vue de face, pas par transparence) et de taille
            # raisonnable. Une homographie « dégénérée » estimée sur des points presque
            # alignés ne respecte pas ces conditions.
            if not forme_plausible(coins):
                continue

            # On ramène le résultat aux pixels de l'image d'origine : H_origine = S^-1 H.
            detections.append(Detection(ref.id, ref.titre, coins / facteur, retour_echelle @ H,
                                        int(votes[i]), inliers, ref.hauteur / ref.largeur))

        detections.sort(key=lambda d: -d.inliers)
        return detections


def forme_plausible(coins):
    """Vrai si le quadrilatère (4 coins, ordre de la référence) est convexe, non retourné et assez grand."""
    # Aire signée (formule du lacet / shoelace). Les coins de la référence tournent dans
    # le sens horaire à l'écran (axe y vers le bas) : l'aire est positive. Une aire négative
    # signifie une image miroir.
    x, y = coins[:, 0], coins[:, 1]
    aire = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if aire < AIRE_MIN:
        return False
    # Convexité : les produits vectoriels des côtés consécutifs ont tous le même signe.
    cotes = np.roll(coins, -1, axis=0) - coins
    produits = cotes[:, 0] * np.roll(cotes, -1, axis=0)[:, 1] - cotes[:, 1] * np.roll(cotes, -1, axis=0)[:, 0]
    return bool((produits > 0).all())
