"""Télécharge les pochettes de référence depuis le Cover Art Archive (via l'API MusicBrainz).

Pour chaque album : recherche du release-group sur MusicBrainz, puis récupération de
l'image « front » sur le Cover Art Archive. Écrit albums/<id>.jpg et albums/albums.json.

Lancement : uv run python outils/telecharger_pochettes.py
"""

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

ALBUMS = [
    ("The Beatles", "Abbey Road"),
    ("Nirvana", "Nevermind"),
    ("Fleetwood Mac", "Rumours"),
    ("Daft Punk", "Random Access Memories"),
    ("Radiohead", "OK Computer"),
]

DOSSIER = Path(__file__).resolve().parent.parent / "albums"
USER_AGENT = "pochettes-ar/0.1 (projet de cours d'informatique graphique)"
TAILLE = "1200"  # 250, 500 ou 1200 px


def requete(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as rep:
        return rep.read()


def identifiant(artiste, album):
    texte = unicodedata.normalize("NFKD", f"{artiste}_{album}").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", texte.lower()).strip("_")


def chercher_release_group(artiste, album):
    query = f'releasegroup:"{album}" AND artist:"{artiste}" AND primarytype:album'
    url = "https://musicbrainz.org/ws/2/release-group/?" + urllib.parse.urlencode(
        {"query": query, "fmt": "json", "limit": 5}
    )
    resultats = json.loads(requete(url))["release-groups"]
    if not resultats:
        raise LookupError(f"aucun release-group pour {artiste} – {album}")
    return resultats[0]


def main():
    DOSSIER.mkdir(exist_ok=True)
    table = {}
    for artiste, album in ALBUMS:
        rg = chercher_release_group(artiste, album)
        time.sleep(1.1)  # limite MusicBrainz : 1 requête par seconde

        cle = identifiant(artiste, album)
        fichier = DOSSIER / f"{cle}.jpg"
        url_image = f"https://coverartarchive.org/release-group/{rg['id']}/front-{TAILLE}"
        fichier.write_bytes(requete(url_image))

        table[cle] = {
            "artiste": artiste,
            "album": album,
            "fichier": fichier.name,
            "musicbrainz_release_group": rg["id"],
            "titre_musicbrainz": rg["title"],
            "source": url_image,
        }
        print(f"{artiste} – {album} -> {fichier.name} ({rg['title']}, {rg['id']})")

    (DOSSIER / "albums.json").write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
