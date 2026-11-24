"""Association album -> objet 3D à rendre.

Trois sources :
0. pochettes inconnues découvertes par la webcam : objet composé par IA à partir d'une
   description de la pochette (pochettes_ar/generation/), ajouté en cours de route ;
1. cache/<id_album>.obj : modèle généré par IA à partir de la pochette (détourage rembg +
   TripoSR + simplification, voir outils/generer_modeles_ia.ipynb, exécuté sur Google Colab) ;
2. modeles/ : nos modèles construits par le code (outils/generer_modeles.py), d'après
   modeles/modeles.json. Ils servent de solution de repli si un modèle IA manque ou est illisible.
"""

import json
from pathlib import Path

from pochettes_ar.rendu.objet import ObjetRendu

RACINE = Path(__file__).resolve().parent.parent.parent

# Entrée « modeles.json » des modèles IA : ils n'ont qu'une partie et pas de pivot propre ;
# l'animation « ia » (scene/animations.py) les fait tourner et flotter d'un bloc.
ENTREE_IA = {"modele": "ia.obj", "ombrage": "phong", "pivots": {}}


class Catalogue:
    def __init__(self, dossier_modeles=RACINE / "modeles", dossier_cache=RACINE / "cache", ia=True):
        self.dossier = Path(dossier_modeles)
        self.cache = Path(dossier_cache)
        table = json.loads((self.dossier / "modeles.json").read_text())
        self.entrees = {k: v for k, v in table.items() if not k.startswith("_")}
        self.ia = ia  # préférer les modèles IA quand ils existent
        self._objets = {}  # (id_album, ia ?) -> ObjetRendu (ou None si illisible)
        self.generes = {}  # id -> (ObjetRendu, entrée d'animation) des pochettes découvertes

    def ajouter_genere(self, id_pochette, objet, entree):
        self.generes[id_pochette] = (objet, entree)

    def chemin_ia(self, id_album):
        return self.cache / f"{id_album}.obj"

    def _utilise_ia(self, id_album):
        return self.ia and self._charger_ia(id_album) is not None

    def _charger_ia(self, id_album):
        cle = (id_album, True)
        if cle not in self._objets:
            chemin = self.chemin_ia(id_album)
            try:
                self._objets[cle] = ObjetRendu.depuis_obj(chemin) if chemin.exists() else None
            except (OSError, ValueError, IndexError) as erreur:  # fichier tronqué ou mal formé
                print(f"modèle IA illisible, repli sur le modèle classique : {chemin} ({erreur})")
                self._objets[cle] = None
        return self._objets[cle]

    def objet(self, id_album):
        """L'objet de l'album (chargé à la première demande), ou None si l'album n'a pas de modèle."""
        if id_album in self.generes:
            return self.generes[id_album][0]
        if self._utilise_ia(id_album):
            return self._charger_ia(id_album)
        entree = self.entrees.get(id_album)
        if entree is None:
            return None
        cle = (id_album, False)
        if cle not in self._objets:
            self._objets[cle] = ObjetRendu.depuis_obj(self.dossier / entree["modele"],
                                                      entree.get("ombrage", "phong"))
        return self._objets[cle]

    def entree(self, id_album):
        """Description (modèle, pivots, ombrage) de l'objet renvoyé par objet(), pour l'animer."""
        if id_album in self.generes:
            return self.generes[id_album][1]
        return ENTREE_IA if self._utilise_ia(id_album) else self.entrees.get(id_album)

    def source(self, id_album):
        """« IA », « classique » ou « composé » : d'où vient l'objet affiché (affichage de contrôle)."""
        if id_album in self.generes:
            return "composé"
        return "IA" if self._utilise_ia(id_album) else "classique"
