"""Découverte d'une pochette inconnue : la repérer dans l'image et la redresser.

La reconnaissance (reconnaissance.py) ne retrouve que des pochettes déjà indexées. Pour une
pochette jamais vue, on ne cherche pas son identité : on cherche sa forme. Une pochette est
un grand quadrilatère (un carré vu en perspective) qui contraste avec le fond.

Deux façons de l'acquérir :
1. automatique : contours de Canny, approximation polygonale (Douglas-Peucker), on garde
   les quadrilatères convexes assez grands et de proportions plausibles, et on attend que le
   même quadrilatère reste stable pendant ~0.7 s (la pochette est tenue devant la caméra,
   pas juste traversée) ;
2. manuelle : l'utilisateur place la pochette dans le cadre guide au centre et appuie sur
   une touche ; le cadre sert alors de quadrilatère.

Dans les deux cas, l'homographie quadrilatère -> carré redresse la pochette (vue de face).
Cette image redressée devient sa référence ORB : elle est ensuite suivie et sa pose estimée
exactement comme les pochettes de albums/.
"""

import cv2
import numpy as np

LARGEUR_ANALYSE = 640  # les contours se cherchent sur une image réduite (plus rapide, moins de bruit)
AIRE_MIN, AIRE_MAX = 0.04, 0.8  # aire du quadrilatère / aire de l'image
PROPORTION_MAX = 1.6  # rapport des côtés opposés moyens (un carré en perspective reste proche de 1)
COTE_REDRESSE = 600  # taille de l'image redressée (comme TAILLE_REFERENCE de reconnaissance.py)


def ordonner_coins(coins):
    """4 points -> ordre de la référence : haut-gauche, haut-droit, bas-droit, bas-gauche.
    Haut-gauche : plus petite somme x + y ; bas-droit : plus grande ; haut-droit : plus
    petite différence y - x ; bas-gauche : plus grande."""
    coins = np.asarray(coins, dtype=np.float32).reshape(4, 2)
    somme, difference = coins.sum(axis=1), coins[:, 1] - coins[:, 0]
    return coins[[np.argmin(somme), np.argmin(difference), np.argmax(somme), np.argmax(difference)]]


def _proportion(coins):
    cotes = np.linalg.norm(np.roll(coins, -1, axis=0) - coins, axis=1)
    largeur, hauteur = (cotes[0] + cotes[2]) / 2, (cotes[1] + cotes[3]) / 2
    return max(largeur, hauteur) / max(min(largeur, hauteur), 1e-6)


def trouver_quadrilatere(image_bgr):
    """Le plus grand quadrilatère plausible de l'image (4 coins ordonnés, en pixels), ou None."""
    facteur = min(1.0, LARGEUR_ANALYSE / image_bgr.shape[1])
    petite = cv2.GaussianBlur(cv2.resize(image_bgr, None, fx=facteur, fy=facteur,
                                         interpolation=cv2.INTER_AREA), (5, 5), 0)
    # Bords (Canny) cherchés dans chaque canal de couleur puis réunis : une pochette bleu clair
    # sur un mur gris a presque la même luminosité que lui, mais pas la même couleur.
    bruts = np.zeros(petite.shape[:2], np.uint8)
    for canal in cv2.split(petite):
        bruts |= cv2.Canny(canal, 40, 120)
    # Bords épaissis, pour refermer les petites coupures du contour de la pochette.
    bords = cv2.dilate(bruts, np.ones((3, 3), np.uint8))
    # Pour le contrôle d'appui : tolérance de ±3 pixels (le quadrilatère simplifié passe à
    # 1 ou 2 pixels du bord réel).
    proches = cv2.dilate(bruts, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(bords, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    aire_image = petite.shape[0] * petite.shape[1]
    meilleur, meilleure_aire = None, 0.0
    for contour in contours:
        if cv2.contourArea(cv2.convexHull(contour)) <= max(meilleure_aire, AIRE_MIN * aire_image):
            continue
        # Deux essais : le contour lui-même, puis son enveloppe convexe (utile quand le dessin
        # de la pochette touche son bord et crée des creux dans le contour).
        for forme in (contour, cv2.convexHull(contour)):
            # Douglas-Peucker : simplifie le contour en ne gardant que les sommets à plus de
            # epsilon de la polyligne simplifiée. Une pochette donne 4 sommets.
            for epsilon in (0.02, 0.04):
                poly = cv2.approxPolyDP(forme, epsilon * cv2.arcLength(forme, True), True)
                aire = cv2.contourArea(poly)
                if (len(poly) != 4 or not cv2.isContourConvex(poly)
                        or not AIRE_MIN * aire_image < aire < AIRE_MAX * aire_image or aire <= meilleure_aire):
                    continue
                coins = ordonner_coins(poly)
                if _proportion(coins) > PROPORTION_MAX:
                    continue  # trop allongé : téléphone, livre, écran…
                # Le quadrilatère doit s'appuyer sur de vrais bords sur presque tout son tour
                # (l'enveloppe convexe d'un amas de formes, elle, traverse le fond).
                if _appui_sur_les_bords(proches, coins) < APPUI_MIN:
                    continue
                meilleur, meilleure_aire = coins, aire
    return None if meilleur is None else meilleur / facteur


APPUI_MIN = 0.85  # part du périmètre du quadrilatère qui doit tomber sur un bord détecté


def _appui_sur_les_bords(bords, coins, pas=2.0):
    """Part des points du périmètre (échantillonnés tous les `pas` pixels) situés sur un bord."""
    points = []
    for a, b in zip(coins, np.roll(coins, -1, axis=0)):
        n = max(2, int(np.linalg.norm(b - a) / pas))
        points.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    points = np.concatenate(points).round().astype(int)
    h, l = bords.shape
    points = points[(points[:, 0] >= 0) & (points[:, 0] < l) & (points[:, 1] >= 0) & (points[:, 1] < h)]
    return float((bords[points[:, 1], points[:, 0]] > 0).mean()) if len(points) else 0.0


def cadre_guide(forme, part=0.6):
    """Carré centré dans l'image (part = côté / plus petite dimension), coins ordonnés."""
    h, l = forme[:2]
    c = part * min(h, l) / 2
    cx, cy = l / 2, h / 2
    return np.float32([[cx - c, cy - c], [cx + c, cy - c], [cx + c, cy + c], [cx - c, cy + c]])


def redresser(image_bgr, coins, cote=COTE_REDRESSE):
    """Vue de face de la pochette : homographie coins -> carré cote x cote (vision, pas rendu 3D)."""
    carre = np.float32([[0, 0], [cote, 0], [cote, cote], [0, cote]])
    H = cv2.getPerspectiveTransform(np.float32(coins), carre)
    return cv2.warpPerspective(image_bgr, H, (cote, cote), flags=cv2.INTER_LINEAR)


class Decouvreur:
    """Suit le quadrilatère candidat d'une image à l'autre et signale quand il est stable."""

    def __init__(self, images_stables=20, tolerance=0.03, pause=4.0):
        """images_stables : nombre d'images consécutives où le quadrilatère doit rester en place ;
        tolerance : déplacement maximal des coins, en fraction de la diagonale de l'image ;
        pause : délai (s) après une acquisition ou un refus, avant de chercher à nouveau."""
        self.images_stables, self.tolerance, self.pause = images_stables, tolerance, pause
        self.candidat, self.compte, self.reprise = None, 0, 0.0

    def mettre_a_jour(self, image_bgr, detections, instant):
        """Coins à acquérir quand un quadrilatère inconnu est resté stable, sinon None."""
        if instant < self.reprise:
            return None
        coins = trouver_quadrilatere(image_bgr)
        # Une pochette déjà reconnue n'est pas « inconnue » : on ignore un quadrilatère dont
        # le centre tombe dans le contour d'une détection.
        if coins is not None:
            centre = tuple(float(v) for v in coins.mean(axis=0))
            if any(cv2.pointPolygonTest(d.coins.astype(np.float32), centre, False) >= 0 for d in detections):
                coins = None
        if coins is None:
            self.candidat, self.compte = None, 0
            return None
        diagonale = float(np.hypot(*image_bgr.shape[:2]))
        if self.candidat is not None and np.abs(coins - self.candidat).max() < self.tolerance * diagonale:
            self.compte += 1
        else:
            self.compte = 1
        self.candidat = coins
        if self.compte >= self.images_stables:
            self.attendre(instant)
            return coins
        return None

    def attendre(self, instant):
        """Suspend la recherche pendant self.pause secondes (après une acquisition ou un refus)."""
        self.candidat, self.compte, self.reprise = None, 0, instant + self.pause

    @property
    def progression(self):
        """Avancement (0 à 1) vers l'acquisition du candidat courant, pour l'affichage."""
        return min(1.0, self.compte / self.images_stables)
