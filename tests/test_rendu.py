"""Tests du moteur de rendu : chaque étape vérifiée sur des cas dont on connaît la réponse."""

import numpy as np
import pytest

from pochettes_ar.rendu import camera
from pochettes_ar.rendu.objet import ObjetRendu
from pochettes_ar.rendu.pipeline import Appel, Pipeline
from pochettes_ar.rendu.rasterisation import rasteriser
from pochettes_ar.rendu.shading import blinn_phong
from pochettes_ar.rendu.textures import echantillonner_bilineaire
from pochettes_ar.rendu.transformations import compose, matrice_normales, rotation, rotation_autour, translation
from pochettes_ar.rendu.zbuffer import Tampons
from pochettes_ar.scene.catalogue import Catalogue
from pochettes_ar.vision import calibration

L, H = 320, 240
K = calibration.approximer(L, H).K


def vers_ecran(points_camera):
    """Points du repère caméra -> (x_px, y_px, z_ndc), w, par nos matrices P et viewport."""
    P = camera.matrice_projection(K, L, H, 0.05, 100)
    clip = np.c_[points_camera, np.ones(len(points_camera))] @ P.T
    ndc = clip[:, :3] / clip[:, 3:]
    return (np.c_[ndc, np.ones(len(ndc))] @ camera.matrice_viewport(L, H).T)[:, :3], clip[:, 3]


# --- Transformations et caméra --------------------------------------------------------


def test_rotation_et_normales():
    R = rotation([1, 2, 3], 0.7)
    np.testing.assert_allclose(R[:3, :3] @ R[:3, :3].T, np.eye(3), atol=1e-12)  # orthogonale
    np.testing.assert_allclose(np.linalg.det(R[:3, :3]), 1)
    np.testing.assert_allclose(matrice_normales(R), R[:3, :3], atol=1e-12)  # rotation : inchangée
    M = rotation_autour([1, 0, 0], [0, 0, 1], np.pi / 2)
    np.testing.assert_allclose(M @ [1, 0, 0, 1], [1, 0, 0, 1], atol=1e-12)  # le pivot ne bouge pas
    np.testing.assert_allclose(M @ [2, 0, 0, 1], [1, 1, 0, 1], atol=1e-12)


def test_projection_et_viewport_egalent_le_stenope():
    """P puis /w puis viewport doit donner exactement u = fx X/Z + cx, v = fy Y/Z + cy."""
    points = np.random.default_rng(0).uniform([-1, -1, 0.5], [1, 1, 10], (100, 3))
    ecran, w = vers_ecran(points)
    attendu = points[:, :2] / points[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]
    np.testing.assert_allclose(ecran[:, :2], attendu, atol=1e-9)
    np.testing.assert_allclose(w, points[:, 2])
    # Profondeur : -1 au plan proche, +1 au plan lointain, croissante entre les deux.
    z, _ = vers_ecran(np.array([[0, 0, 0.05], [0, 0, 1], [0, 0, 100]]))
    np.testing.assert_allclose(z[[0, 2], 2], [-1, 1])
    assert -1 < z[1, 2] < 1


# --- Rasterisation et z-buffer -----------------------------------------------------------


def test_triangle_couvre_son_aire():
    tampons = Tampons(L, H)
    ecran = np.array([[[20.0, 30.0], [220.0, 50.0], [90.0, 200.0]]])
    rasteriser(tampons, ecran, np.zeros((1, 3)), np.ones((1, 3)))
    couverts = (tampons.triangle == 0).sum()
    aire = 0.5 * abs((200 * 170) - (70 * 20))
    assert abs(couverts - aire) < 0.02 * aire  # le nombre de centres de pixels couverts ≈ aire
    b = tampons.bary[tampons.triangle == 0]
    assert (b >= -1e-9).all() and (b.sum(axis=1) <= 1 + 1e-9).all()


def test_zbuffer_independant_de_l_ordre():
    """Deux triangles qui se recouvrent : le plus proche gagne, quel que soit l'ordre de dessin."""
    ecran = np.array([[[10.0, 10.0], [300.0, 10.0], [10.0, 230.0]], [[300.0, 230.0], [10.0, 230.0], [300.0, 10.0]]])
    prof = np.array([[0.2] * 3, [0.5] * 3])  # triangle 0 plus proche
    resultats = []
    for ordre in ([0, 1], [1, 0]):
        tampons = Tampons(L, H)
        rasteriser(tampons, ecran[ordre], prof[ordre], np.ones((2, 3)))
        resultats.append(np.array(ordre)[tampons.triangle[tampons.triangle >= 0]])
        assert tampons.profondeur[50, 50] == 0.2 and tampons.profondeur[200, 280] == 0.5
    np.testing.assert_array_equal(*resultats)


def test_interpolation_corrigee_de_la_perspective():
    """Pour un triangle très incliné, la position 3D interpolée en chaque pixel avec les
    barycentriques corrigées doit se reprojeter exactement au centre de ce pixel.
    (Avec des barycentriques non corrigées, l'erreur atteint ~120 px sur ce triangle.)"""
    sommets = np.array([[-1.0, -0.5, 1.0], [1.0, -0.5, 1.0], [0.0, 0.6, 8.0]])
    ecran, w = vers_ecran(sommets)
    tampons = Tampons(L, H)
    rasteriser(tampons, ecran[None, :, :2], ecran[None, :, 2], 1 / w[None])
    ys, xs = np.nonzero(tampons.triangle == 0)
    b = tampons.bary[ys, xs]
    bary = np.c_[b, 1 - b.sum(axis=1)]
    reprojete, _ = vers_ecran(bary @ sommets)
    np.testing.assert_allclose(reprojete[:, 0], xs, atol=1e-6)
    np.testing.assert_allclose(reprojete[:, 1], ys, atol=1e-6)


# --- Shading et textures ---------------------------------------------------------------


def test_blinn_phong():
    n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    l = np.array([0.0, 0.0, 1.0])
    v = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [np.sin(1.0), 0.0, np.cos(1.0)]])
    kd = np.full((3, 3), 0.4)
    ks = np.full((3, 3), 0.2)
    c = blinn_phong(n, l, v, kd, ks, np.full(3, 10.0), ambiante=0.25)
    np.testing.assert_allclose(c[0], 0.25 * 0.4 + 0.4 + 0.2)  # lumière et œil dans l'axe : reflet maximal
    np.testing.assert_allclose(c[2], 0.25 * 0.4 + 0.4 + 0.2 * np.cos(0.5) ** 10)  # h à mi-angle
    dos = blinn_phong(n[:1], -l, v[:1], kd[:1], ks[:1], np.full(1, 10.0), ambiante=0.25)
    np.testing.assert_allclose(dos[0], 0.25 * 0.4)  # face à contre-jour : ambiante seule


def test_echantillonnage_bilineaire():
    texture = np.arange(4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)  # 4 lignes, 3 colonnes
    # centre du texel (ligne 0, colonne 1) : u = 1.5 / 3, v = 1 - 0.5 / 4 (v = 0 en bas)
    np.testing.assert_allclose(echantillonner_bilineaire(texture, np.array([[1.5 / 3, 1 - 0.5 / 4]]))[0], texture[0, 1])
    # à mi-chemin entre les colonnes 0 et 1 de la ligne 2 : moyenne des deux
    milieu = echantillonner_bilineaire(texture, np.array([[1.0 / 3, 1 - 2.5 / 4]]))[0]
    np.testing.assert_allclose(milieu, (texture[2, 0] + texture[2, 1]) / 2)


# --- Pipeline complet ------------------------------------------------------------------


def carre(sens=1):
    """Carré unité dans le plan y = 0, normale +Y (sens = 1) ou -Y (sens = -1)."""
    tri = np.array([[0, 2, 1], [0, 3, 2]]) if sens == 1 else np.array([[0, 1, 2], [0, 2, 3]])
    return ObjetRendu(
        sommets=np.array([[-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5]]),
        normales=np.tile([0.0, sens, 0.0], (4, 1)), uv=np.zeros((4, 2)), triangles=tri,
        partie_sommet=np.zeros(4, int), noms_parties=["carre"], materiau_triangle=np.zeros(2, int),
        kd=np.array([[1.0, 0.0, 0.0]]), ks=np.zeros((1, 3)), ns=np.ones(1), textures=[None])


# Pochette vue de face à 2 largeurs : Y (normale de la pochette) vers la caméra.
VUE_FACE = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 2.0], [0, 0, 0, 1]], dtype=np.float64)


def test_faces_arriere_eliminees():
    pipeline = Pipeline(L, H, K)
    fond = np.zeros((H, L, 3), np.uint8)
    pipeline.rendre(fond, [Appel(carre(+1), VUE_FACE)])
    assert pipeline.stats["triangles"] == 2
    # côté du carré à l'écran : f x 1 / 2 pixels -> aire ≈ (f / 2)²
    assert abs(pipeline.stats["pixels"] - (K[0, 0] / 2) ** 2) < 0.03 * (K[0, 0] / 2) ** 2
    pipeline.rendre(fond, [Appel(carre(-1), VUE_FACE)])
    assert pipeline.stats["triangles"] == 0 and pipeline.stats["pixels"] == 0


def test_animation_par_partie():
    """Une matrice modèle par partie : déplacer le carré hors du champ le fait disparaître."""
    pipeline = Pipeline(L, H, K)
    fond = np.zeros((H, L, 3), np.uint8)
    image = pipeline.rendre(fond, [Appel(carre(), VUE_FACE, {"carre": translation(5, 0, 0)})])
    assert pipeline.stats["pixels"] == 0 and (image == fond).all()


@pytest.mark.parametrize("album", Catalogue().entrees)
def test_rendu_des_modeles(album):
    """Chaque modèle se rend sans erreur, vu en plongée, et couvre une partie raisonnable de l'image."""
    objet = Catalogue().objet(album)
    vue = compose(VUE_FACE, rotation([1, 0, 0], np.radians(-40)))
    pipeline = Pipeline(L, H, K)
    fond = np.full((H, L, 3), 128, np.uint8)
    image = pipeline.rendre(fond, [Appel(objet, vue)])
    assert 0.005 * L * H < pipeline.stats["pixels"] < 0.5 * L * H
    assert (image != fond).any(axis=2).sum() <= pipeline.stats["pixels"]  # seuls les pixels rendus changent
