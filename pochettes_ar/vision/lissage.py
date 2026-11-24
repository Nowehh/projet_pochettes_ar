"""Lissage temporel de la pose, pour que l'objet 3D ne tremble pas.

D'une image à l'autre, les coins détectés bougent de quelques dixièmes de pixel (bruit du
capteur, points ORB légèrement différents, RANSAC). solvePnP transmet ce bruit à la pose, et
l'objet, plus haut que la pochette, l'amplifie : il « tremble ». On filtre donc la pose dans
le temps, séparément pour chaque album :

1. Moyenne exponentielle : pose_lissée <- (1 - a) pose_lissée + a pose_mesurée.
   - Translation : moyenne ordinaire des vecteurs.
   - Rotation : on ne peut pas moyenner deux matrices de rotation coefficient par coefficient
     (le résultat n'est plus une rotation). On passe par les quaternions unitaires, où une
     interpolation linéaire suivie d'une renormalisation (« nlerp ») donne une rotation
     intermédiaire, très proche de l'interpolation sphérique (slerp) pour de petits angles.
2. Coefficient a adaptatif : petit (lissage fort) quand la pose bouge peu, c'est du bruit ;
   proche de 1 quand elle bouge beaucoup, c'est un vrai mouvement, qu'on suit sans retard.
3. Rejet des sauts : l'ambiguïté des cibles planes (voir pose.py) fait parfois basculer la
   pose d'un coup, sur une seule image. Un changement de rotation brutal n'est adopté que
   s'il dure plusieurs images d'affilée ; sinon on garde la pose précédente.
4. Oubli : une pochette perdue trop longtemps repart de zéro quand on la retrouve.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pochettes_ar.vision.pose import Pose


# --- Quaternions unitaires q = (w, x, y, z) : rotation d'angle a autour de l'axe unitaire k ----
#     q = (cos(a/2), sin(a/2) k). q et -q représentent la même rotation.


def quaternion_depuis_matrice(R):
    """Méthode de Shepperd : on part du plus grand des quatre termes pour rester précis."""
    trace = np.trace(R)
    if trace > 0:
        s = 2 * np.sqrt(1 + trace)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, s / 4, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, s / 4, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, s / 4]
    q = np.array(q)
    return q / np.linalg.norm(q)


def matrice_depuis_quaternion(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def angle_entre(q1, q2):
    """Angle (radians) de la rotation qui fait passer de q1 à q2."""
    return 2 * np.arccos(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))


@dataclass
class _Etat:
    q: np.ndarray  # rotation lissée
    t: np.ndarray  # translation lissée
    derniere_vue: float  # instant de la dernière mesure (secondes)
    erreur: float
    sauts: int = 0  # nombre d'images consécutives en désaccord brutal avec la pose lissée


class LisseurPoses:
    def __init__(self, a_min=0.15, a_max=0.9, mouvement_t=0.05, mouvement_angle=np.radians(6),
                 seuil_saut=np.radians(35), confirmation=3, oubli=0.5):
        """
        a_min, a_max      : coefficient de la moyenne exponentielle, au repos et en mouvement ;
        mouvement_t       : écart de translation, relatif à la distance, à partir duquel la
                            pochette est considérée en mouvement (a = a_max) ;
        mouvement_angle   : idem pour l'écart de rotation ;
        seuil_saut        : écart de rotation au-delà duquel une mesure est suspecte ;
        confirmation      : nombre d'images suspectes d'affilée avant d'adopter la nouvelle pose ;
        oubli             : délai (s) au-delà duquel une pochette perdue repart de zéro.
        """
        self.a_min, self.a_max = a_min, a_max
        self.mouvement_t, self.mouvement_angle = mouvement_t, mouvement_angle
        self.seuil_saut, self.confirmation, self.oubli = seuil_saut, confirmation, oubli
        self.etats = {}

    def lisser(self, id_album, pose, instant):
        """Pose lissée de l'album id_album, à partir de la pose mesurée à l'instant donné."""
        q, t = quaternion_depuis_matrice(pose.R), np.asarray(pose.tvec, dtype=np.float64)
        etat = self.etats.get(id_album)
        if etat is None or instant - etat.derniere_vue > self.oubli:
            self.etats[id_album] = _Etat(q, t, instant, pose.erreur)
            return pose
        etat.derniere_vue = instant

        # q et -q sont la même rotation : on prend le représentant du même côté que la pose
        # lissée, sinon la moyenne passerait par le « mauvais chemin » (le plus long).
        if np.dot(q, etat.q) < 0:
            q = -q
        angle = angle_entre(q, etat.q)

        # 3. Rejet des sauts : on attend que le changement se confirme.
        if angle > self.seuil_saut:
            etat.sauts += 1
            if etat.sauts < self.confirmation:
                return self._pose(etat)
            etat.q, etat.t, etat.erreur, etat.sauts = q, t, pose.erreur, 0  # confirmé : on l'adopte
            return self._pose(etat)
        etat.sauts = 0

        # 2. Coefficient adaptatif, de a_min (immobile) à a_max (mouvement franc).
        ecart_t = np.linalg.norm(t - etat.t) / np.linalg.norm(etat.t)
        mouvement = min(1.0, max(ecart_t / self.mouvement_t, angle / self.mouvement_angle))
        a = self.a_min + (self.a_max - self.a_min) * mouvement

        # 1. Moyenne exponentielle (nlerp pour la rotation).
        etat.t = (1 - a) * etat.t + a * t
        etat.q = (1 - a) * etat.q + a * q
        etat.q /= np.linalg.norm(etat.q)
        etat.erreur = pose.erreur
        return self._pose(etat)

    @staticmethod
    def _pose(etat):
        R = matrice_depuis_quaternion(etat.q)
        return Pose(cv2.Rodrigues(R)[0].ravel(), etat.t.copy(), R, etat.erreur)
