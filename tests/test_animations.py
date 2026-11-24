"""Animations : chaque album s'anime sans erreur, et les mouvements ont le sens physique voulu."""

import numpy as np
import pytest

from pochettes_ar.rendu.pipeline import Appel, Pipeline
from pochettes_ar.scene.animations import RAYON_ROUE, VITESSE_VOITURE, animer
from pochettes_ar.scene.catalogue import Catalogue
from tests.test_rendu import H, K, L, VUE_FACE, carre

CATALOGUE = Catalogue(ia=False)  # animations propres aux modèles classiques


def point(M, p):
    return (M @ [*p, 1.0])[:3]


@pytest.mark.parametrize("album", CATALOGUE.entrees)
@pytest.mark.parametrize("t", [0.0, 0.37, 5.0, 61.2])
def test_etat_anime_valide(album, t):
    objet, entree = CATALOGUE.objet(album), CATALOGUE.entrees[album]
    etat = animer(entree, objet, t)
    assert set(etat.get("matrices_parties", {})) <= set(objet.noms_parties)
    for M in etat.get("matrices_parties", {}).values():
        # animation rigide : rotation (orthogonale, det 1) + translation, sans déformation
        np.testing.assert_allclose(M[:3, :3] @ M[:3, :3].T, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(M[3], [0, 0, 0, 1])
    if "sommets" in etat:
        assert etat["sommets"].shape == objet.sommets.shape
        np.testing.assert_allclose(np.linalg.norm(etat["normales"], axis=1), 1, atol=1e-9)
    # et l'objet animé se rend
    Pipeline(L, H, K).rendre(np.zeros((H, L, 3), np.uint8), [Appel(objet, VUE_FACE, **etat)])


def test_roues_roulent_sans_glisser():
    """Le point bas d'une roue est immobile par rapport au sol (roulement sans glissement)."""
    album = "the_beatles_abbey_road"
    objet, entree = CATALOGUE.objet(album), CATALOGUE.entrees[album]
    centre = np.array(entree["pivots"]["roue_av_g"]["centre"])
    t, dt = 1.0, 1e-4
    M = animer(entree, objet, t)["matrices_parties"]["roue_av_g"]
    # point de la roue (repère du modèle) qui touche la pochette à l'instant t
    bas = point(np.linalg.inv(M), point(M, centre) - [0, RAYON_ROUE, 0])
    a = point(M, bas)
    b = point(animer(entree, objet, t + dt)["matrices_parties"]["roue_av_g"], bas)
    assert np.linalg.norm(b - a) / dt < 0.01 * VITESSE_VOITURE
    # alors que le moyeu avance à la vitesse de la voiture, vers +X
    a = point(animer(entree, objet, t)["matrices_parties"]["roue_av_g"], centre)
    b = point(animer(entree, objet, t + dt)["matrices_parties"]["roue_av_g"], centre)
    np.testing.assert_allclose((b - a) / dt, [VITESSE_VOITURE, 0, 0], atol=1e-6)


def test_pendule_garde_sa_longueur():
    album = "fleetwood_mac_rumours"
    objet, entree = CATALOGUE.objet(album), CATALOGUE.entrees[album]
    centre = np.array(entree["pivots"]["boule_mate"]["centre"])
    boule = np.array([centre[0], 0.25, 0.0])
    for t in np.linspace(0, 3, 7):
        M = animer(entree, objet, t)["matrices_parties"]["boule_mate"]
        np.testing.assert_allclose(np.linalg.norm(point(M, boule) - centre), np.linalg.norm(boule - centre))


def test_normales_du_billet_perpendiculaires():
    """Les normales recalculées sont perpendiculaires à la surface déformée."""
    album = "nirvana_nevermind"
    objet, entree = CATALOGUE.objet(album), CATALOGUE.entrees[album]
    etat = animer(entree, objet, 0.8)
    s, n = etat["sommets"], etat["normales"]
    billet = objet.noms_parties.index("billet")
    tri = objet.triangles[(objet.partie_sommet[objet.triangles] == billet).all(axis=1)]
    a, b, c = s[tri[:, 0]], s[tri[:, 1]], s[tri[:, 2]]
    assert np.abs(s[objet.partie_sommet == billet, 2] - objet.sommets[objet.partie_sommet == billet, 2]).max() > 0.005
    for i, j in ((0, 1), (0, 2)):
        arete = (b - a) if j == 1 else (c - a)
        arete /= np.linalg.norm(arete, axis=1, keepdims=True)
        # normale moyenne du triangle (sommets voisins) : quasi perpendiculaire à ses arêtes
        nm = n[tri].mean(axis=1)
        assert np.abs((nm * arete).sum(axis=1)).max() < 0.1


def test_decalage_uv_fait_defiler_la_texture():
    """Un décalage d'une demi-période sur une texture à deux bandes inverse les couleurs."""
    objet = carre()
    objet.uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    objet.kd = np.ones((1, 3))
    objet.textures = [np.zeros((1, 2, 3), np.float32)]
    objet.textures[0][0, 0] = [1, 0, 0]  # moitié gauche rouge, moitié droite bleue
    objet.textures[0][0, 1] = [0, 0, 1]
    pipeline = Pipeline(L, H, K)
    fond = np.zeros((H, L, 3), np.uint8)
    x_gauche = int(K[0, 2] - K[0, 0] / 8)  # quart gauche du carré
    y = int(K[1, 2])
    avant = pipeline.rendre(fond, [Appel(objet, VUE_FACE)])[y, x_gauche]
    apres = pipeline.rendre(fond, [Appel(objet, VUE_FACE, decalages_uv={"carre": (0.5, 0.0)})])[y, x_gauche]
    assert avant[2] > avant[0] and apres[0] > apres[2]  # BGR : rouge, puis bleu
