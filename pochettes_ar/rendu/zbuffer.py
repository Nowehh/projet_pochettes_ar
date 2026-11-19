"""Tampons d'image du rendu (M&S 8.2.3).

- profondeur : le z-buffer. Pour chaque pixel, la profondeur (z_ndc) du fragment le plus
  proche dessiné jusqu'ici. Un nouveau fragment n'est accepté que s'il est plus proche :
  la visibilité est réglée pixel par pixel, quel que soit l'ordre des triangles.
- triangle : numéro du triangle visible en ce pixel (-1 : aucun) ;
- bary : ses coordonnées barycentriques (b0, b1) corrigées de la perspective (b2 = 1 - b0 - b1).

Ces deux derniers tampons servent au shading différé : on n'éclaire chaque pixel qu'une
fois, pour le seul triangle finalement visible, au lieu d'éclairer chaque fragment testé.
"""

import numpy as np


class Tampons:
    def __init__(self, largeur, hauteur):
        self.largeur, self.hauteur = largeur, hauteur
        self.profondeur = np.empty((hauteur, largeur), np.float64)
        self.triangle = np.empty((hauteur, largeur), np.int64)
        self.bary = np.empty((hauteur, largeur, 2), np.float64)
        self.effacer()

    def effacer(self):
        self.profondeur.fill(np.inf)  # « infiniment loin » : le premier fragment gagne toujours
        self.triangle.fill(-1)
