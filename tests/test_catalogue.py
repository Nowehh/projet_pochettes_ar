"""Catalogue : modèle IA (cache/) en priorité, modèle classique en repli ; couleurs par sommet."""

import numpy as np

from pochettes_ar.rendu.pipeline import Appel, Pipeline
from pochettes_ar.scene.animations import animer
from pochettes_ar.scene.catalogue import Catalogue
from pochettes_ar.scene.maillage import charger_obj
from tests.test_rendu import H, K, L, VUE_FACE

ALBUM = "nirvana_nevermind"

# Tétraèdre au format écrit par le notebook Colab : « v x y z r g b », faces sans normales.
OBJ_IA = """mtllib {nom}.mtl
o ia
usemtl ia
v -0.2 0 0.2 1 0 0
v 0.2 0 0.2 1 0 0
v 0 0 -0.2 1 0 0
v 0 0.3 0 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
"""


def ecrire_modele_ia(dossier, id_album=ALBUM, contenu=OBJ_IA):
    dossier.mkdir(exist_ok=True)
    (dossier / f"{id_album}.obj").write_text(contenu.format(nom=id_album))
    (dossier / f"{id_album}.mtl").write_text("newmtl ia\nKd 1 1 1\nKs 0 0 0\nNs 10\n")


def test_couleurs_par_sommet_lues(tmp_path):
    ecrire_modele_ia(tmp_path)
    m = charger_obj(tmp_path / f"{ALBUM}.obj")[0].maillage
    assert len(m.triangles) == 4
    rouges = m.sommets[:, 1] == 0
    np.testing.assert_allclose(m.couleurs[rouges], [[1, 0, 0]] * rouges.sum())
    np.testing.assert_allclose(m.couleurs[~rouges], [[0, 0, 1]])
    # sans couleur dans le fichier : blanc (seul le matériau compte)
    sans = charger_obj(Catalogue().dossier / "casque.obj")[0].maillage
    np.testing.assert_allclose(sans.couleurs, 1)


def test_modele_ia_prefere_puis_repli(tmp_path):
    ecrire_modele_ia(tmp_path)
    catalogue = Catalogue(dossier_cache=tmp_path)
    assert catalogue.source(ALBUM) == "IA"
    assert catalogue.objet(ALBUM).noms_parties == ["ia"]
    assert catalogue.source("fleetwood_mac_rumours") == "classique"  # pas de modèle IA : repli
    catalogue.ia = False  # touche g
    assert catalogue.source(ALBUM) == "classique"
    assert "billet" in catalogue.objet(ALBUM).noms_parties


def test_modele_ia_illisible_donne_le_repli(tmp_path):
    ecrire_modele_ia(tmp_path, contenu="v 0 0 0\nf 1 2 3\n")  # indices hors limites
    catalogue = Catalogue(dossier_cache=tmp_path)
    assert catalogue.source(ALBUM) == "classique"
    assert "billet" in catalogue.objet(ALBUM).noms_parties


def test_modele_ia_anime_et_rendu_en_couleur(tmp_path):
    ecrire_modele_ia(tmp_path)
    catalogue = Catalogue(dossier_cache=tmp_path)
    objet = catalogue.objet(ALBUM)
    pipeline = Pipeline(L, H, K)
    fond = np.zeros((H, L, 3), np.uint8)
    image = pipeline.rendre(fond, [Appel(objet, VUE_FACE, **animer(catalogue.entree(ALBUM), objet, 0.4))])
    assert pipeline.stats["pixels"] > 100
    couverts = (image != fond).any(axis=2)
    bleu, rouge = image[couverts][:, 0].astype(int), image[couverts][:, 2].astype(int)
    assert (bleu > rouge).any() and (rouge > bleu).any()  # les deux couleurs, interpolées
