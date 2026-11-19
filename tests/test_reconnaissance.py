"""Reconnaissance sur des scènes synthétiques : une pochette déformée en perspective,
incrustée sur un fond texturé, avec bruit, flou et changement d'éclairage. On connaît
les vrais coins, donc on peut mesurer l'erreur de localisation."""

import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from pochettes_ar.vision.reconnaissance import Reconnaisseur

ALBUMS = Path(__file__).resolve().parent.parent / "albums"
TABLE = json.loads((ALBUMS / "albums.json").read_text())
LARGEUR, HAUTEUR = 1280, 720


@pytest.fixture(scope="module")
def reconnaisseur():
    return Reconnaisseur(ALBUMS)


def fond(rng):
    """Fond « bureau » : bruit lissé + rectangles et cercles aléatoires."""
    img = cv2.GaussianBlur(rng.integers(0, 255, (HAUTEUR, LARGEUR, 3), dtype=np.uint8), (0, 0), 6)
    for _ in range(40):
        couleur = rng.integers(0, 255, 3).tolist()
        x, y = int(rng.integers(0, LARGEUR)), int(rng.integers(0, HAUTEUR))
        if rng.random() < 0.5:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(20, 200)), y + int(rng.integers(20, 200))), couleur, -1)
        else:
            cv2.circle(img, (x, y), int(rng.integers(10, 80)), couleur, -1)
    return img


def incruster(scene, id_album, coins, lumiere=1.0):
    pochette = cv2.imread(str(ALBUMS / TABLE[id_album]["fichier"]))
    h, w = pochette.shape[:2]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(coins))
    deformee = cv2.warpPerspective(pochette, H, (LARGEUR, HAUTEUR))
    masque = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (LARGEUR, HAUTEUR)) > 0
    scene[masque] = np.clip(deformee[masque] * lumiere, 0, 255).astype(np.uint8)


def degrader(scene, rng):
    scene = cv2.GaussianBlur(scene, (0, 0), 1.0)
    return np.clip(scene + rng.normal(0, 6, scene.shape), 0, 255).astype(np.uint8)


# Vues variées : de face, inclinée, tournée, petite et lointaine.
VUES = {
    "face": [[440, 160], [840, 160], [840, 560], [440, 560]],
    "perspective": [[380, 200], [820, 150], [860, 620], [400, 560]],
    "tournee": [[640, 110], [930, 360], [660, 650], [370, 390]],
    "lointaine": [[900, 420], [1080, 430], [1070, 610], [890, 600]],
}


@pytest.mark.parametrize("vue", VUES)
@pytest.mark.parametrize("id_album", TABLE)
def test_reconnait_chaque_pochette(reconnaisseur, id_album, vue):
    rng = np.random.default_rng(zlib.crc32(f"{id_album}-{vue}".encode()))
    scene = fond(rng)
    incruster(scene, id_album, VUES[vue], lumiere=0.75)
    detections = reconnaisseur.reconnaitre(degrader(scene, rng))

    assert detections, f"aucune détection ; stats : {reconnaisseur.stats}"
    assert [d.id for d in detections] == [id_album]
    # Les coins sont extrapolés par l'homographie (ce ne sont pas des points ORB) :
    # l'erreur tolérée est relative à la taille de la pochette dans l'image.
    vrais = np.float32(VUES[vue])
    cote = np.linalg.norm(vrais - np.roll(vrais, 1, axis=0), axis=1).mean()
    erreur = np.linalg.norm(detections[0].coins - vrais, axis=1).max()
    assert erreur < 0.04 * cote, f"coins à {erreur:.1f} px des vrais coins (côté {cote:.0f} px)"


def test_deux_pochettes_a_la_fois(reconnaisseur):
    rng = np.random.default_rng(0)
    scene = fond(rng)
    incruster(scene, "nirvana_nevermind", [[80, 120], [480, 140], [470, 540], [70, 520]])
    incruster(scene, "radiohead_ok_computer", [[760, 160], [1160, 130], [1180, 540], [780, 560]])
    detections = reconnaisseur.reconnaitre(degrader(scene, rng))
    assert {d.id for d in detections} == {"nirvana_nevermind", "radiohead_ok_computer"}


def test_rien_sur_un_fond_sans_pochette(reconnaisseur):
    for graine in range(5):
        rng = np.random.default_rng(100 + graine)
        assert reconnaisseur.reconnaitre(degrader(fond(rng), rng)) == []


def test_image_miroir_rejetee(reconnaisseur):
    """Une pochette vue en miroir n'est pas physiquement possible : elle doit être rejetée."""
    rng = np.random.default_rng(7)
    scene = fond(rng)
    incruster(scene, "the_beatles_abbey_road", [[840, 160], [440, 160], [440, 560], [840, 560]])
    assert reconnaisseur.reconnaitre(degrader(scene, rng)) == []
