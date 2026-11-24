"""Étape 1 de la génération : un modèle de vision décrit la pochette en quelques mots.

Entrée : la pochette redressée (vue de face, carrée). Sortie : ce qui est écrit dessus
(artiste, album, si c'est lisible) et une description très courte du sujet principal, pensée
pour être modélisée ensuite avec des formes simples (étape 2, composition.py).
"""

from dataclasses import dataclass

import cv2

from pochettes_ar.generation import ollama

# Modèles de vision acceptés, par ordre de préférence (le premier installé est utilisé).
# Variantes « instruct » d'abord : les variantes « thinking » écrivent un long raisonnement caché
# avant de répondre (40 s au lieu de quelques secondes), même avec think = False.
MODELES_VISION = ["qwen3-vl:8b-instruct", "qwen2.5vl:7b", "qwen3-vl:8b", "llava:7b"]

CONSIGNE = """Tu regardes la pochette d'un album de musique. Réponds en JSON avec quatre champs :
- "sujet" : en 3 à 8 mots, en français, le sujet principal représenté sur l'image : un objet,
  un animal, un personnage ou un élément concret, facile à reconstruire en 3D avec des formes
  simples (exemples : « une voiture blanche », « un bébé qui nage vers un billet », « un
  prisme qui décompose la lumière »). Pas de style, pas d'ambiance, pas le texte de la
  pochette. Si l'image est abstraite, propose l'objet concret qui l'évoque le mieux.
- "texte_artiste" : le nom de l'artiste s'il est écrit sur la pochette, sinon "".
- "texte_album" : le titre de l'album s'il est écrit sur la pochette, sinon "".
  Recopie exactement, n'invente rien, ne mélange pas les deux.
- "couleurs" : les 3 couleurs dominantes, en hexadécimal ("#rrggbb")."""

SCHEMA = {
    "type": "object",
    "properties": {
        "sujet": {"type": "string"},
        "texte_artiste": {"type": "string"},
        "texte_album": {"type": "string"},
        "couleurs": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sujet", "texte_artiste", "texte_album", "couleurs"],
}


@dataclass
class Description:
    artiste: str
    album: str
    description: str
    couleurs: list  # chaînes « #rrggbb »
    modele: str  # modèle d'IA qui a produit la description

    @property
    def titre(self):
        if self.artiste and self.album:
            return f"{self.artiste} - {self.album}"
        return self.album or self.artiste or self.description


def decrire(image_bgr, modele=None, taille=512):
    """Description de la pochette (image BGR carrée). Lève OllamaIndisponible si l'IA ne répond pas."""
    modele = modele or ollama.choisir(MODELES_VISION)
    if modele is None:
        raise ollama.OllamaIndisponible(f"aucun modèle de vision installé parmi {MODELES_VISION}")
    image = cv2.resize(image_bgr, (taille, taille), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("encodage JPEG impossible")
    reponse = ollama.discuter(modele, CONSIGNE, images=[jpeg.tobytes()], schema=SCHEMA, temperature=0.0)
    return Description(
        artiste=str(reponse.get("texte_artiste", "")).strip(),
        album=str(reponse.get("texte_album", "")).strip(),
        description=str(reponse.get("sujet", "")).strip() or "un disque vinyle",
        couleurs=[c for c in reponse.get("couleurs", []) if isinstance(c, str)][:3],
        modele=modele,
    )
