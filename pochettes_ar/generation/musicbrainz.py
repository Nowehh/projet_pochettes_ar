"""Confirmation facultative de l'artiste et de l'album lus sur la pochette, via MusicBrainz.

Le modèle de vision recopie le texte de la pochette, parfois avec des fautes ou des
majuscules approximatives. Une recherche dans MusicBrainz (base de données musicale ouverte)
redonne l'orthographe officielle. En cas d'échec (pas de réseau, rien trouvé, score faible),
on garde simplement ce qu'a lu l'IA.
"""

import json
import urllib.parse
import urllib.request

USER_AGENT = "pochettes-ar/0.2 (projet de cours d'informatique graphique)"
SCORE_MIN = 90  # score de pertinence MusicBrainz (0-100)


def confirmer(artiste, album, timeout=5.0):
    """(artiste, album, mbid du release-group) officiels, ou None."""
    if not artiste or not album:
        return None
    requete = f'releasegroup:"{album}" AND artist:"{artiste}"'
    url = "https://musicbrainz.org/ws/2/release-group/?" + urllib.parse.urlencode(
        {"query": requete, "fmt": "json", "limit": 1})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as reponse:
            resultats = json.loads(reponse.read()).get("release-groups", [])
    except (OSError, ValueError):
        return None
    if not resultats or int(resultats[0].get("score", 0)) < SCORE_MIN:
        return None
    rg = resultats[0]
    credit = "".join(c.get("name", "") + c.get("joinphrase", "") for c in rg.get("artist-credit", []))
    return credit or artiste, rg.get("title", album), rg.get("id")
