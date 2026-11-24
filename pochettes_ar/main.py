"""Point d'entrée : boucle capture -> reconnaissance -> pose -> rendu -> affichage.

Pour chaque pochette reconnue : pose (solvePnP), lissée dans le temps (vision/lissage.py),
puis rendu de son modèle 3D animé (scene/animations.py) par notre pipeline
(pochettes_ar/rendu/), composé par-dessus l'image webcam.

Lancement :
    uv run python -m pochettes_ar.main                  # webcam 0
    uv run python -m pochettes_ar.main --camera 1
    uv run python -m pochettes_ar.main --image photo.jpg   # sur une photo
Touches : q ou Échap pour quitter ; i : statistiques ; a : axes X, Y, Z ; c : contour et titre ;
m : modèle 3D (affiché / masqué) ; l : lissage de la pose ; p : pause de l'animation ;
g : modèles générés par IA (cache/) ou modèles classiques (modeles/).
"""

import argparse
import time
from pathlib import Path

import cv2

from pochettes_ar.rendu.pipeline import Appel, Pipeline
from pochettes_ar.scene.animations import animer
from pochettes_ar.scene.catalogue import Catalogue
from pochettes_ar.vision import calibration
from pochettes_ar.vision.affichage import dessiner_axes, dessiner_detection, dessiner_infos
from pochettes_ar.vision.capture import ouvrir_camera
from pochettes_ar.vision.lissage import LisseurPoses
from pochettes_ar.vision.pose import estimer_pose
from pochettes_ar.vision.reconnaissance import Reconnaisseur

RACINE = Path(__file__).resolve().parent.parent
FENETRE = "Pochettes AR"


class Application:
    def __init__(self):
        self.reconnaisseur = Reconnaisseur(RACINE / "albums")
        self.catalogue = Catalogue(RACINE / "modeles", RACINE / "cache")
        self.lisseur = LisseurPoses()
        self.calib = self.pipeline = None
        self.options = {"infos": True, "axes": False, "contour": True, "modele": True,
                        "lissage": True, "pause": False}
        self.horloge = time.perf_counter()  # instant réel de la dernière image
        self.t_animation = 0.0  # temps de l'animation (s'arrête en pause)

    def traiter(self, image, ips=0.0):
        maintenant = time.perf_counter()
        if not self.options["pause"]:
            self.t_animation += maintenant - self.horloge
        self.horloge = maintenant

        if self.pipeline is None:  # la résolution réelle n'est connue qu'à la première image
            self.calib = calibration.charger(image.shape[1], image.shape[0])
            self.pipeline = Pipeline(image.shape[1], image.shape[0], self.calib.K)

        detections = self.reconnaisseur.reconnaitre(image)
        poses = [estimer_pose(d, self.calib) for d in detections]
        if self.options["lissage"]:
            poses = [self.lisseur.lisser(d.id, pose, maintenant) if pose is not None else None
                     for d, pose in zip(detections, poses)]

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
          + ", ".join(f"{r.titre} ({r.nb_points} pts)" for r in app.reconnaisseur.references))
    print("modèles : " + ", ".join(f"{r.titre} ({app.catalogue.source(r.id)})" for r in app.reconnaisseur.references))

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"image illisible : {args.image}")
        cv2.imshow(FENETRE, app.traiter(image))
        cv2.waitKey(0)
        return

    camera = ouvrir_camera(args.camera)
    ips, precedent = 0.0, time.perf_counter()
    try:
        while True:
            ok, image = camera.read()
            if not ok:
                break
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
            for lettre, option in (("i", "infos"), ("a", "axes"), ("c", "contour"), ("m", "modele"),
                                   ("l", "lissage"), ("p", "pause")):
                if touche == ord(lettre):
                    app.options[option] = not app.options[option]
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
