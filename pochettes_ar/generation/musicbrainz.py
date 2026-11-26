"""Confirmation facultative de l'artiste et de l'album lus sur la pochette, via MusicBrainz.

Le modèle de vision recopie le texte de la pochette, parfois avec des fautes ou des
majuscules approximatives. Une recherche dans MusicBrainz (base de données musicale ouverte)
redonne l'orthographe officielle. En cas d'échec (pas de réseau, rien trouvé, score faible),
on garde simplement ce qu'a lu l'IA.
"""

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

USER_AGENT = "pochettes-ar/0.2 (projet de cours d'informatique graphique)"
SCORE_MIN = 90  # score de pertinence MusicBrainz (0-100)

# MusicBrainz n'accepte qu'une requête par seconde et par client : au-delà, il répond
# « 503 Service Temporarily Unavailable ». Toutes nos requêtes passent donc par _obtenir(),
# qui les espace (verrou partagé entre les fils d'exécution) et réessaie après un 503.
INTERVALLE = 1.1  # secondes entre deux requêtes
_verrou = threading.Lock()
_derniere = [0.0]


def _obtenir(url, accept, timeout, essais=3):
    for essai in range(essais):
        with _verrou:
            attente = _derniere[0] + INTERVALLE - time.monotonic()
            if attente > 0:
                time.sleep(attente)
            _derniere[0] = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
            with urllib.request.urlopen(req, timeout=timeout) as reponse:  # suit les redirections
                return reponse.read()
        except urllib.error.HTTPError as erreur:
            if erreur.code != 503 or essai == essais - 1:
                raise


def confirmer(artiste, album, timeout=5.0):
    """(artiste, album, mbid du release-group) officiels, ou None."""
    if not artiste or not album:
        return None
    requete = f'releasegroup:"{album}" AND artist:"{artiste}"'
    url = "https://musicbrainz.org/ws/2/release-group/?" + urllib.parse.urlencode(
        {"query": requete, "fmt": "json", "limit": 1})
    try:
        resultats = json.loads(_obtenir(url, "application/json", timeout)).get("release-groups", [])
    except (OSError, ValueError):
        return None
    if not resultats or int(resultats[0].get("score", 0)) < SCORE_MIN:
        return None
    rg = resultats[0]
    credit = "".join(c.get("name", "") + c.get("joinphrase", "") for c in rg.get("artist-credit", []))
    return credit or artiste, rg.get("title", album), rg.get("id")


def pochette_officielle(mbid, taille=500, timeout=10.0):
    """Image (BGR) de la pochette officielle du release-group, depuis le Cover Art Archive,
    ou None (pas de réseau, pas de pochette enregistrée)."""
    url = f"https://coverartarchive.org/release-group/{mbid}/front-{taille}"
    try:
        octets = _obtenir(url, "image/*", timeout)
    except OSError:
        return None
    return cv2.imdecode(np.frombuffer(octets, np.uint8), cv2.IMREAD_COLOR)
