"""Génération d'un objet 3D pour une pochette inconnue, en arrière-plan, avec cache sur disque.

    pochette redressée
      -> 1. description (modèle de vision)           : « un prisme qui décompose la lumière »
      -> 2. identification de l'album (texte lu, ou album reconnu par l'IA, vérifié avec
            MusicBrainz et la pochette officielle ; voir identification.py)
      -> 3. composition (modèle de langage)          : pièces simples en JSON
      -> 4. construction (notre code)                : maillage, puis rendu par notre pipeline

Les étapes 1 et 3 prennent plusieurs secondes : elles tournent dans un fil d'exécution à part,
pour que la boucle webcam continue (la pochette est suivie et affiche « génération en cours »
pendant ce temps). Une seule génération à la fois : le Mac n'a qu'un GPU à partager.

Si l'IA est absente ou échoue, l'objet de repli (platine vinyle) est utilisé.

Cache : cache/pochettes/<id>/reference.jpg (image redressée, la référence ORB) et
pochette.json (titre, description, composition). Au lancement suivant, la pochette est
reconnue directement et son objet reconstruit sans appeler l'IA.
"""

import json
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from pochettes_ar.generation import composition as compo
from pochettes_ar.generation.construction import construire
from pochettes_ar.generation.description import decrire
from pochettes_ar.generation.identification import identifier
from pochettes_ar.generation.ollama import OllamaIndisponible


@dataclass
class Resultat:
    id: str
    titre: str
    infos: dict  # contenu de pochette.json
    objet: object = None  # ObjetRendu
    entree: dict = field(default_factory=dict)  # entrée d'animation
    erreur: str = ""  # message si l'IA a échoué (objet de repli)


def generer(image_bgr, decrire_=decrire, composer_=compo.composer, identifier_=identifier):
    """Pipeline complet, synchrone. Renvoie (infos, erreur) ; ne lève pas d'exception."""
    infos = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "titre": "Pochette inconnue"}
    try:
        d = decrire_(image_bgr)
        infos.update(artiste=d.artiste, album=d.album, description=d.description,
                     couleurs=d.couleurs, modele_vision=d.modele, titre=d.titre)
        infos.update(reconnu_artiste=d.reconnu_artiste, reconnu_album=d.reconnu_album)
        identite = identifier_(d, image_bgr)
        if identite:
            infos.update(artiste=identite.artiste, album=identite.album, titre=identite.titre,
                         musicbrainz_release_group=identite.mbid, identification=identite.methode)
        infos["composition"] = composer_(d.description, d.couleurs)
        return infos, ""
    except (OllamaIndisponible, ValueError, KeyError, TypeError) as erreur:
        infos["composition"] = compo.VINYLE
        return infos, f"{type(erreur).__name__} : {erreur}"


def _resultat(id_pochette, infos, erreur=""):
    objet, entree = construire(compo.valider(infos["composition"]))
    return Resultat(id_pochette, infos.get("titre", "Pochette inconnue"), infos, objet, entree, erreur)


class Generateur:
    def __init__(self, dossier_cache, generer_=generer):
        self.dossier = Path(dossier_cache) / "pochettes"
        self._generer = generer_
        self._executeur = ThreadPoolExecutor(max_workers=1, thread_name_prefix="generation")
        self._termines = queue.Queue()
        self.en_cours = set()

    def nouvel_id(self):
        return time.strftime("pochette_%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"

    def lancer(self, id_pochette, image_redressee):
        """Démarre la génération en arrière-plan ; le résultat arrivera par termines()."""
        self.en_cours.add(id_pochette)
        dossier = self.dossier / id_pochette
        dossier.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dossier / "reference.jpg"), image_redressee)

        def travail():
            try:
                infos, erreur = self._generer(image_redressee)
                if erreur:
                    infos["erreur"] = erreur
                (dossier / "pochette.json").write_text(json.dumps(infos, indent=2, ensure_ascii=False) + "\n")
                self._termines.put(_resultat(id_pochette, infos, erreur))
            except Exception as erreur:  # dernier filet : la boucle webcam ne doit jamais attendre
                infos = {"titre": "Pochette inconnue", "composition": compo.VINYLE, "erreur": repr(erreur)}
                self._termines.put(_resultat(id_pochette, infos, repr(erreur)))

        self._executeur.submit(travail)

    def termines(self):
        """Résultats arrivés depuis le dernier appel (à appeler à chaque image, sans attendre)."""
        resultats = []
        while True:
            try:
                r = self._termines.get_nowait()
            except queue.Empty:
                return resultats
            self.en_cours.discard(r.id)
            resultats.append(r)

    def charger_cache(self):
        """Pochettes générées lors des lancements précédents : [(Resultat, image de référence)]."""
        trouves = []
        for dossier in sorted(self.dossier.glob("*/")) if self.dossier.exists() else []:
            fichier, image = dossier / "pochette.json", cv2.imread(str(dossier / "reference.jpg"))
            if not fichier.exists() or image is None:
                continue  # génération interrompue avant la fin
            try:
                trouves.append((_resultat(dossier.name, json.loads(fichier.read_text())), image))
            except (ValueError, KeyError) as erreur:
                print(f"cache illisible, ignoré : {dossier} ({erreur})")
        return trouves

    def arreter(self):
        self._executeur.shutdown(wait=False, cancel_futures=True)
