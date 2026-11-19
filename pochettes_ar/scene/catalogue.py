"""Association album -> objet 3D à rendre, d'après modeles/modeles.json."""

import json
from pathlib import Path

from pochettes_ar.rendu.objet import ObjetRendu


class Catalogue:
    def __init__(self, dossier_modeles="modeles"):
        self.dossier = Path(dossier_modeles)
        table = json.loads((self.dossier / "modeles.json").read_text())
        self.entrees = {k: v for k, v in table.items() if not k.startswith("_")}
        self._objets = {}

    def objet(self, id_album):
        """L'objet de l'album (chargé à la première demande), ou None si l'album n'a pas de modèle."""
        entree = self.entrees.get(id_album)
        if entree is None:
            return None
        if id_album not in self._objets:
            self._objets[id_album] = ObjetRendu.depuis_obj(self.dossier / entree["modele"],
                                                           entree.get("ombrage", "phong"))
        return self._objets[id_album]
