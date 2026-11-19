"""Vérifie les modèles générés : relecture OBJ, normales, orientation des triangles."""

import json
from pathlib import Path

import numpy as np
import pytest

from outils import generer_modeles
from pochettes_ar.scene.maillage import charger_obj, ecrire_obj

MODELES = Path(__file__).resolve().parent.parent / "modeles"


@pytest.fixture(scope="module", params=list(generer_modeles.MODELES))
def modele(request):
    nom, fabrique = generer_modeles.MODELES[request.param]
    return nom, fabrique()[0]


def test_aller_retour_obj(modele, tmp_path):
    nom, parties = modele
    ecrire_obj(tmp_path / f"{nom}.obj", parties)
    relues = charger_obj(tmp_path / f"{nom}.obj")
    assert [p.nom for p in relues] == [p.nom for p in parties]
    for p, r in zip(parties, relues):
        assert r.materiau == p.materiau
        assert len(r.maillage.triangles) == len(p.maillage.triangles)
        # mêmes triangles, à la précision d'écriture près
        np.testing.assert_allclose(r.maillage.sommets[r.maillage.triangles],
                                   p.maillage.sommets[p.maillage.triangles], atol=1e-6)
        np.testing.assert_allclose(r.maillage.uv[r.maillage.triangles],
                                   p.maillage.uv[p.maillage.triangles], atol=1e-6)


def test_normales_et_orientation(modele):
    """Normales unitaires et du même côté que la normale géométrique (b - a) x (c - a)."""
    for p in modele[1]:
        m = p.maillage
        np.testing.assert_allclose(np.linalg.norm(m.normales, axis=1), 1, atol=1e-6)
        t = m.sommets[m.triangles]
        geometrique = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        assert (np.linalg.norm(geometrique, axis=1) > 1e-12).all(), f"{p.nom} : triangle dégénéré"
        moyenne = m.normales[m.triangles].sum(axis=1)
        assert (np.einsum("ij,ij->i", geometrique, moyenne) > 0).all(), f"{p.nom} : triangle retourné"
        assert ((m.uv >= -1e-9) & (m.uv <= 1 + 1e-9)).all()


def test_objets_poses_sur_la_pochette(modele):
    sommets = np.concatenate([p.maillage.sommets for p in modele[1]])
    assert sommets[:, 1].min() >= -1e-9  # rien sous la pochette
    assert np.abs(sommets[:, [0, 2]]).max() <= 0.5  # rien ne dépasse du carré


def test_fichiers_generes_a_jour():
    catalogue = json.loads((MODELES / "modeles.json").read_text())
    for album, (nom, _) in generer_modeles.MODELES.items():
        assert catalogue[album]["modele"] == f"{nom}.obj"
        parties = charger_obj(MODELES / f"{nom}.obj")
        assert {p.nom for p in parties} == set(catalogue[album]["parties"])
        for p in parties:
            if p.materiau.texture:
                assert (MODELES / p.materiau.texture).exists()
