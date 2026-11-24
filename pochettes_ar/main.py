"""Point d'entrée : boucle capture -> reconnaissance -> pose -> rendu -> affichage.

Pour chaque pochette reconnue : pose (solvePnP), lissée dans le temps (vision/lissage.py),
puis rendu de son modèle 3D animé (scene/animations.py) par notre pipeline
(pochettes_ar/rendu/), composé par-dessus l'image webcam.

Pochettes inconnues (vision/decouverte.py, pochettes_ar/generation/) : un grand
quadrilatère stable qui n'est pas une pochette connue est redressé et devient une nouvelle
référence ; en arrière-plan, une IA locale (Ollama) le décrit en quelques mots puis compose
un objet 3D simple, que notre code construit. La pochette est suivie pendant la génération.

Lancement :
    uv run python -m pochettes_ar.main                  # webcam 0
    uv run python -m pochettes_ar.main --camera 1
    uv run python -m pochettes_ar.main --image photo.jpg   # sur une photo
Touches : q ou Échap pour quitter ; i : statistiques ; a : axes X, Y, Z ; c : contour et titre ;
m : modèle 3D (affiché / masqué) ; l : lissage de la pose ; p : pause de l'animation ;
g : modèles générés par IA (cache/) ou modèles classiques (modeles/) ;
d : découverte automatique des pochettes inconnues ; n : capturer la pochette du cadre guide.
"""

import argparse
import time
from pathlib import Path

import cv2

from pochettes_ar.generation import ollama
from pochettes_ar.generation.composition import MODELES_TEXTE
from pochettes_ar.generation.description import MODELES_VISION
from pochettes_ar.generation.generateur import Generateur
from pochettes_ar.rendu.pipeline import Appel, Pipeline
from pochettes_ar.scene.animations import animer
from pochettes_ar.scene.catalogue import Catalogue
from pochettes_ar.vision import calibration
from pochettes_ar.vision.affichage import (dessiner_axes, dessiner_candidat, dessiner_detection,
                                           dessiner_guide, dessiner_infos)
from pochettes_ar.vision.capture import ouvrir_camera
from pochettes_ar.vision.decouverte import Decouvreur, cadre_guide, redresser, trouver_quadrilatere
from pochettes_ar.vision.lissage import LisseurPoses
from pochettes_ar.vision.pose import estimer_pose
from pochettes_ar.vision.reconnaissance import Reconnaisseur

RACINE = Path(__file__).resolve().parent.parent
FENETRE = "Pochettes AR"
TITRE_EN_COURS = "Nouvelle pochette (generation en cours...)"


class Application:
    def __init__(self):
        self.reconnaisseur = Reconnaisseur(RACINE / "albums")
        self.catalogue = Catalogue(RACINE / "modeles", RACINE / "cache")
        self.lisseur = LisseurPoses()
        self.decouvreur = Decouvreur()
        self.generateur = Generateur(RACINE / "cache")
        self.calib = self.pipeline = None
        self.options = {"infos": True, "axes": False, "contour": True, "modele": True,
                        "lissage": True, "pause": False, "decouverte": True}
        self.horloge = time.perf_counter()  # instant réel de la dernière image
        self.t_animation = 0.0  # temps de l'animation (s'arrête en pause)
        self.candidat = None  # quadrilatère inconnu en cours d'observation (affichage)

        # Pochettes découvertes lors des lancements précédents : reconnues et animées sans IA.
        for resultat, reference in self.generateur.charger_cache():
            if self.reconnaisseur.ajouter_reference(resultat.id, resultat.titre, reference):
                self.catalogue.ajouter_genere(resultat.id, resultat.objet, resultat.entree)

    # --- Pochettes inconnues -----------------------------------------------------------------

    def acquerir(self, image, coins, instant=0.0):
        """Redresse la pochette délimitée par coins et lance la génération de son objet.
        Renvoie l'identifiant de la nouvelle pochette, ou None si elle est refusée."""
        self.decouvreur.attendre(instant)
        pochette = redresser(image, coins)
        if self.reconnaisseur.reconnaitre(pochette):
            return None  # c'est une pochette déjà connue, vue de trop près ou de biais
        id_pochette = self.generateur.nouvel_id()
        if not self.reconnaisseur.ajouter_reference(id_pochette, TITRE_EN_COURS, pochette):
            print("pochette refusée : pas assez de texture pour la suivre (ORB)")
            return None
        self.generateur.lancer(id_pochette, pochette)
        print(f"nouvelle pochette {id_pochette} : génération de son objet en arrière-plan")
        return id_pochette

    def _recevoir_generations(self):
        for r in self.generateur.termines():
            self.reconnaisseur.renommer(r.id, r.titre)
            self.catalogue.ajouter_genere(r.id, r.objet, r.entree)
            if r.erreur:
                print(f"{r.id} : IA indisponible ou en échec ({r.erreur}) -> objet de repli")
            else:
                print(f"{r.id} : {r.titre} -> « {r.infos.get('description')} » -> "
                      f"{r.infos['composition']['nom']} ({len(r.infos['composition']['pieces'])} pièces)")

    # --- Une image ------------------------------------------------------------------------------

    def traiter(self, image, ips=0.0):
        maintenant = time.perf_counter()
        if not self.options["pause"]:
            self.t_animation += maintenant - self.horloge
        self.horloge = maintenant

        if self.pipeline is None:  # la résolution réelle n'est connue qu'à la première image
            self.calib = calibration.charger(image.shape[1], image.shape[0])
            self.pipeline = Pipeline(image.shape[1], image.shape[0], self.calib.K)

        self._recevoir_generations()
        detections = self.reconnaisseur.reconnaitre(image)
        poses = [estimer_pose(d, self.calib) for d in detections]
        if self.options["lissage"]:
            poses = [self.lisseur.lisser(d.id, pose, maintenant) if pose is not None else None
                     for d, pose in zip(detections, poses)]

        # Découverte : un quadrilatère inconnu resté stable est acquis.
        self.candidat = None
        if self.options["decouverte"]:
            coins = self.decouvreur.mettre_a_jour(image, detections, maintenant)
            if coins is not None:
                self.acquerir(image, coins, maintenant)
            self.candidat = self.decouvreur.candidat

        # Rendu 3D : un appel de dessin par pochette reconnue qui a une pose et un modèle.
        if self.options["modele"]:
            appels = []
            for d, pose in zip(detections, poses):
                objet = self.catalogue.objet(d.id)
                if pose is not None and objet is not None:
                    etat = animer(self.catalogue.entree(d.id), objet, self.t_animation)
                    appels.append(Appel(objet, pose.matrice(), **etat))
            image = self.pipeline.rendre(image, appels)

        # Annotations 2D de contrôle, par-dessus le rendu.
        for detection, pose in zip(detections, poses):
            if self.options["contour"]:
                dessiner_detection(image, detection, pose)
            if self.options["axes"] and pose is not None:
                dessiner_axes(image, pose, self.calib.K)
        if self.candidat is not None:
            dessiner_candidat(image, self.candidat, self.decouvreur.progression)
        elif not detections:
            dessiner_guide(image, cadre_guide(image.shape))
        if self.options["infos"]:
            dessiner_infos(image, self.reconnaisseur.stats, ips, self.calib,
                           self.pipeline.stats if self.options["modele"] else None)
        return image


def main():
    parser = argparse.ArgumentParser(description="Reconnaissance de pochettes d'album")
    parser.add_argument("--camera", type=int, default=0, help="indice de la webcam")
    parser.add_argument("--image", help="analyser une image fixe au lieu de la webcam")
    args = parser.parse_args()

    app = Application()
    print(f"{len(app.reconnaisseur.references)} pochettes indexées : "
          + ", ".join(f"{r.titre} ({app.catalogue.source(r.id)})" for r in app.reconnaisseur.references))
    installes = ollama.modeles_installes()
    vision, texte = ollama.choisir(MODELES_VISION, installes), ollama.choisir(MODELES_TEXTE, installes)
    if vision and texte:
        print(f"IA locale (Ollama) : description par {vision}, composition par {texte}")
    else:
        print("IA locale indisponible (Ollama arrêté ou modèles absents) : "
              "les pochettes inconnues recevront l'objet de repli (platine vinyle)")

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"image illisible : {args.image}")
        # Photo fixe : pas d'attente de stabilité, une pochette inconnue est acquise tout de suite.
        if not app.reconnaisseur.reconnaitre(image):
            coins = trouver_quadrilatere(image)
            app.acquerir(image, coins if coins is not None else cadre_guide(image.shape))
        app.options["decouverte"] = False
        while True:
            cv2.imshow(FENETRE, app.traiter(image.copy()))
            if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                break
        return

    camera = ouvrir_camera(args.camera)
    ips, precedent = 0.0, time.perf_counter()
    try:
        while True:
            ok, image = camera.read()
            if not ok:
                break
            brute = image.copy()  # avant les annotations, pour une capture propre (touche n)
            cv2.imshow(FENETRE, app.traiter(image, ips))

            maintenant = time.perf_counter()
            ips = 0.9 * ips + 0.1 / max(maintenant - precedent, 1e-6)  # moyenne glissante
            precedent = maintenant

            touche = cv2.waitKey(1) & 0xFF
            if touche in (ord("q"), 27):
                break
            if touche == ord("g"):
                app.catalogue.ia = not app.catalogue.ia
                print("modèles", "IA (cache/) quand ils existent" if app.catalogue.ia else "classiques")
            if touche == ord("n"):
                if app.acquerir(brute, cadre_guide(brute.shape), maintenant) is None:
                    print("capture refusée : pochette déjà connue, ou pas assez de texture")
            for lettre, option in (("i", "infos"), ("a", "axes"), ("c", "contour"), ("m", "modele"),
                                   ("l", "lissage"), ("p", "pause"), ("d", "decouverte")):
                if touche == ord(lettre):
                    app.options[option] = not app.options[option]
    finally:
        camera.release()
        app.generateur.arreter()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
