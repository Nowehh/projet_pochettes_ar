"""Pochettes inconnues : découverte, composition validée, construction, génération en arrière-plan.

L'IA (Ollama) est remplacée par des réponses fixes : ces tests tournent sans elle. Un test
d'intégration avec la vraie IA est en fin de fichier, ignoré si Ollama n'est pas disponible.
"""

import time

import cv2
import numpy as np
import pytest

from pochettes_ar.generation import composition as compo
from pochettes_ar.generation import ollama
from pochettes_ar.generation.construction import construire
from pochettes_ar.generation.description import MODELES_VISION, Description
from pochettes_ar.generation.generateur import Generateur, generer
from pochettes_ar.generation.identification import identifier, meme_pochette
from pochettes_ar.rendu.pipeline import Appel, Pipeline
from pochettes_ar.scene.animations import animer
from pochettes_ar.vision.decouverte import Decouvreur, cadre_guide, ordonner_coins, redresser, trouver_quadrilatere
from pochettes_ar.vision.reconnaissance import Reconnaisseur
from tests import test_reconnaissance as T
from tests.test_rendu import H, K, L, VUE_FACE

VOITURE = {"nom": "voiture", "animation": "rebondit", "pieces": [
    {"forme": "boite", "position": [0, 2, 0], "taille": [8, 2, 4], "rotation": [0, 0, 0],
     "couleur": "#d42a2a", "brillance": 0.6, "tourne_sur_soi": "aucun"},
    {"forme": "cylindre", "position": [2.6, 1.1, 2.1], "taille": [2.2, 0.8, 2.2], "rotation": [90, 0, 0],
     "couleur": "#111", "brillance": 0.1, "tourne_sur_soi": "z"},
]}


# --- Composition : toute réponse de l'IA est remise dans les bornes -------------------------


def test_valider_corrige_une_reponse_absurde():
    c = compo.valider({"animation": "danse", "pieces": [
        {"forme": "pyramide", "position": [1e9, "a"], "taille": [-3, 0, float("nan")], "couleur": 12,
         "brillance": 7, "tourne_sur_soi": "w"},
        "pas une pièce",
    ]})
    assert c["animation"] == "tourne" and len(c["pieces"]) == 1
    p = c["pieces"][0]
    assert p["forme"] == "boite" and p["position"] == [20, 0, 0]
    assert p["taille"] == [0.05, 0.05, 1] and p["brillance"] == 1 and p["tourne_sur_soi"] == "aucun"
    with pytest.raises(ValueError):
        compo.valider({"pieces": []})
    with pytest.raises(ValueError):
        compo.valider("du texte")


def test_couleurs():
    assert compo.couleur_rgb("#ff8000") == (1.0, 128 / 255, 0.0)
    assert compo.couleur_rgb("0f0") == (0.0, 1.0, 0.0)
    assert compo.couleur_rgb("rouge") == (0.8, 0.8, 0.8)


# --- Construction : composition -> objet posé sur la pochette, à la bonne taille ------------


@pytest.mark.parametrize("composition", [VOITURE, compo.VINYLE])
def test_construction_dans_le_repere_du_projet(composition):
    objet, entree = construire(compo.valider(composition))
    mini, maxi = objet.sommets.min(axis=0), objet.sommets.max(axis=0)
    np.testing.assert_allclose((maxi - mini).max(), 0.6)  # plus grande dimension = TAILLE
    np.testing.assert_allclose(mini[1], 0, atol=1e-12)  # posé sur la pochette
    np.testing.assert_allclose([(mini[0] + maxi[0]) / 2, (mini[2] + maxi[2]) / 2], 0, atol=1e-12)
    # Toutes les faces sont orientées vers l'extérieur : volume signé positif pour chaque pièce.
    for i in range(len(objet.noms_parties)):
        tri = objet.triangles[(objet.partie_sommet[objet.triangles] == i).all(axis=1)]
        a, b, c = (objet.sommets[tri[:, k]] for k in range(3))
        assert np.einsum("ij,ij->i", a, np.cross(b, c)).sum() > 0
    # Les pièces qui tournent sur elles-mêmes ont un pivot, au centre de la pièce.
    for nom, pivot in entree["pivots"].items():
        sommets = objet.sommets[objet.partie_sommet == objet.noms_parties.index(nom)]
        np.testing.assert_allclose(pivot["centre"], (sommets.min(0) + sommets.max(0)) / 2, atol=1e-9)


def test_roue_tourne_autour_de_son_axe():
    objet, entree = construire(compo.valider(VOITURE))
    pivot = entree["pivots"]["piece_1"]
    np.testing.assert_allclose(np.abs(pivot["axe"]), [0, 0, 1], atol=1e-9)  # cylindre couché : axe z
    centre = np.array(pivot["centre"])
    for t in (0.0, 0.7):
        M = animer(entree, objet, t)["matrices_parties"]["piece_1"]
        # La roue tourne sur elle-même : son centre ne bouge pas (sauf le rebond, vertical)…
        np.testing.assert_allclose((M[:3, :3] @ centre + M[:3, 3] - centre)[[0, 2]], 0, atol=1e-12)
        # … et sa rotation garde l'axe z fixe.
        np.testing.assert_allclose(M[:3, :3] @ [0, 0, 1], [0, 0, 1], atol=1e-12)


def test_objet_compose_se_rend():
    objet, entree = construire(compo.valider(VOITURE))
    pipeline = Pipeline(L, H, K)
    pipeline.rendre(np.zeros((H, L, 3), np.uint8), [Appel(objet, VUE_FACE, **animer(entree, objet, 1.0))])
    assert pipeline.stats["pixels"] > 500


# --- Découverte d'une pochette inconnue dans l'image ------------------------------------------


COINS = np.float32([[420, 180], [820, 200], [800, 590], [400, 560]])


def scene_avec_pochette(graine=0, coins=COINS):
    rng = np.random.default_rng(graine)
    scene = T.fond(rng)
    T.incruster(scene, "fleetwood_mac_rumours", coins)
    return scene


def test_ordonner_coins():
    melanges = COINS[[2, 0, 3, 1]]
    np.testing.assert_allclose(ordonner_coins(melanges), COINS)


def test_trouve_et_redresse_la_pochette():
    scene = scene_avec_pochette()
    coins = trouver_quadrilatere(scene)
    assert coins is not None
    assert np.abs(coins - COINS).max() < 12  # à quelques pixels près (contours lissés)
    pochette = redresser(scene, coins)
    assert pochette.shape == (600, 600, 3)
    # La pochette redressée ressemble à l'originale (corrélation des niveaux de gris).
    originale = cv2.resize(cv2.imread(str(T.ALBUMS / "fleetwood_mac_rumours.jpg")), (600, 600))
    a, b = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(float).ravel() for x in (pochette, originale))
    assert np.corrcoef(a, b)[0, 1] > 0.8


def test_decouvreur_attend_la_stabilite_et_ignore_les_pochettes_connues():
    scene = scene_avec_pochette()
    decouvreur = Decouvreur(images_stables=5, pause=1.0)
    resultats = [decouvreur.mettre_a_jour(scene, [], k / 30) for k in range(5)]
    assert all(r is None for r in resultats[:4]) and resultats[4] is not None
    assert decouvreur.mettre_a_jour(scene, [], 5 / 30) is None  # en pause après l'acquisition
    # Une pochette reconnue n'est pas « inconnue ».
    class Connue:
        coins = COINS
    decouvreur = Decouvreur(images_stables=1)
    assert decouvreur.mettre_a_jour(scene, [Connue()], 0.0) is None


def test_cadre_guide_centre():
    coins = cadre_guide((720, 1280, 3))
    np.testing.assert_allclose(coins.mean(axis=0), [640, 360])
    np.testing.assert_allclose(coins[1, 0] - coins[0, 0], 0.6 * 720)


def test_pochette_ajoutee_puis_reconnue(tmp_path):
    """Une pochette redressée ajoutée comme référence est ensuite reconnue dans une autre vue."""
    reconnaisseur = Reconnaisseur(T.ALBUMS)
    # Pochette « inconnue » : une image synthétique texturée, absente de albums/
    rng = np.random.default_rng(5)
    inconnue = cv2.GaussianBlur(rng.integers(0, 255, (600, 600, 3), dtype=np.uint8), (0, 0), 3)
    for _ in range(40):
        cv2.circle(inconnue, tuple(int(v) for v in rng.integers(0, 600, 2)), int(rng.integers(10, 60)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    assert reconnaisseur.ajouter_reference("pochette_test", "Inconnue", inconnue)
    scene = T.fond(rng)
    coins = np.float32([[300, 150], [700, 180], [690, 560], [280, 540]])
    H_ = cv2.getPerspectiveTransform(np.float32([[0, 0], [600, 0], [600, 600], [0, 600]]), coins)
    masque = cv2.warpPerspective(np.full((600, 600), 255, np.uint8), H_, scene.shape[1::-1]) > 0
    scene[masque] = cv2.warpPerspective(inconnue, H_, scene.shape[1::-1])[masque]
    detections = reconnaisseur.reconnaitre(scene)
    assert detections and detections[0].id == "pochette_test"
    assert np.abs(detections[0].coins - coins).max() < 5
    reconnaisseur.renommer("pochette_test", "Artiste - Album")
    assert reconnaisseur.reconnaitre(scene)[0].titre == "Artiste - Album"


# --- Génération en arrière-plan ------------------------------------------------------------------


def fausse_description(image):
    return Description("Artiste", "Album", "une voiture rouge", ["#d42a2a"], "faux-modele")


def test_generer_avec_ia_simulee():
    infos, erreur = generer(np.zeros((8, 8, 3), np.uint8), decrire_=fausse_description,
                            composer_=lambda d, c: compo.valider(VOITURE), identifier_=lambda d, i: None)
    assert erreur == "" and infos["titre"] == "Artiste - Album"
    assert infos["description"] == "une voiture rouge" and infos["composition"]["nom"] == "voiture"


def test_generer_sans_ia_donne_le_repli():
    def absente(image):
        raise ollama.OllamaIndisponible("Ollama ne répond pas")
    infos, erreur = generer(np.zeros((8, 8, 3), np.uint8), decrire_=absente)
    assert "OllamaIndisponible" in erreur and infos["composition"] == compo.VINYLE


def test_generateur_en_arriere_plan_et_cache(tmp_path):
    def lente(image):
        time.sleep(0.2)
        return {"titre": "Artiste - Album", "description": "une voiture", "composition": VOITURE}, ""
    generateur = Generateur(tmp_path, generer_=lente)
    image = np.full((600, 600, 3), 100, np.uint8)
    debut = time.perf_counter()
    generateur.lancer("p1", image)
    assert time.perf_counter() - debut < 0.1  # lancer() n'attend pas la génération
    assert "p1" in generateur.en_cours and generateur.termines() == []
    for _ in range(100):
        resultats = generateur.termines()
        if resultats:
            break
        time.sleep(0.02)
    assert resultats[0].id == "p1" and resultats[0].titre == "Artiste - Album"
    assert resultats[0].objet is not None and "p1" not in generateur.en_cours
    # Au lancement suivant : la pochette et son objet reviennent du cache, sans IA.
    (cache,) = Generateur(tmp_path).charger_cache()
    assert cache[0].id == "p1" and cache[0].infos["composition"]["nom"] == "voiture"
    assert cache[1].shape == (600, 600, 3)
    generateur.arreter()


# --- Intégration avec la vraie IA (ignoré si Ollama ou ses modèles sont absents) ---------------


@pytest.mark.skipif(ollama.choisir(MODELES_VISION) is None or ollama.choisir(compo.MODELES_TEXTE) is None,
                    reason="Ollama ou ses modèles ne sont pas disponibles")
@pytest.mark.lent
def test_generation_reelle():
    image = cv2.imread(str(T.ALBUMS / "the_beatles_abbey_road.jpg"))
    infos, erreur = generer(image)
    assert infos["titre"] == "The Beatles - Abbey Road", infos  # reconnue sans texte, vérifiée par l'image
    assert erreur == "", erreur
    construire(compo.valider(infos["composition"]))


# --- Identification de l'album -------------------------------------------------------------------


def album(nom):
    return cv2.imread(str(T.ALBUMS / f"{nom}.jpg"))


def description(texte=("", ""), reconnu=("", "")):
    return Description(*texte, "sujet", [], "faux-modele", *reconnu)


def faux_musicbrainz(artiste, album_):
    base = {("The Beatles", "Abbey Road"): ("The Beatles", "Abbey Road", "mbid-abbey"),
            ("NIRVANA", "NEVERMIND"): ("Nirvana", "Nevermind", "mbid-nevermind")}
    return base.get((artiste, album_))


def test_identification_par_le_texte():
    i = identifier(description(texte=("NIRVANA", "NEVERMIND")), album("nirvana_nevermind"),
                   confirmer_=faux_musicbrainz, telecharger_=lambda m: pytest.fail("pas besoin de l'image"))
    assert i.titre == "Nirvana - Nevermind" and i.methode == "texte"


def test_album_reconnu_et_verifie_par_l_image():
    """Vue de la webcam (perspective, bruit) comparée à la pochette officielle : acceptée."""
    vue = redresser(scene_avec_pochette(), trouver_quadrilatere(scene_avec_pochette()))
    officielle = album("fleetwood_mac_rumours")
    ok, inliers = meme_pochette(vue, officielle)
    assert ok and inliers > 50
    i = identifier(description(reconnu=("The Beatles", "Abbey Road")), album("the_beatles_abbey_road"),
                   confirmer_=faux_musicbrainz, telecharger_=lambda m: album("the_beatles_abbey_road"))
    assert i.titre == "The Beatles - Abbey Road" and "image" in i.methode


def test_hallucination_rejetee():
    """L'IA « reconnaît » Abbey Road sur la pochette de Rumours : la pochette officielle
    d'Abbey Road ne correspond pas à l'image, le nom est refusé."""
    i = identifier(description(reconnu=("The Beatles", "Abbey Road")), album("fleetwood_mac_rumours"),
                   confirmer_=faux_musicbrainz, telecharger_=lambda m: album("the_beatles_abbey_road"))
    assert i is None
    for a, b in [("nirvana_nevermind", "the_beatles_abbey_road"), ("radiohead_ok_computer", "daft_punk_random_access_memories")]:
        assert not meme_pochette(album(a), album(b))[0]


def test_album_inconnu_de_musicbrainz_ou_sans_reseau():
    assert identifier(description(reconnu=("Personne", "Rien")), album("nirvana_nevermind"),
                      confirmer_=faux_musicbrainz) is None
    assert identifier(description(reconnu=("The Beatles", "Abbey Road")), album("the_beatles_abbey_road"),
                      confirmer_=faux_musicbrainz, telecharger_=lambda m: None) is None
