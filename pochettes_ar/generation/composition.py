"""Étape 2 de la génération : un modèle de langage compose un objet 3D très simple.

À partir de la description courte (étape 1), le modèle répond par un assemblage de formes
simples (pavés, sphères, cylindres, cônes), chacune avec sa position, sa taille, sa rotation
et sa couleur, au format JSON imposé par SCHEMA. Il ne produit aucune géométrie lui-même :
c'est notre code (construction.py, avec scene/primitives.py) qui fabrique les maillages.

Le modèle peut se tromper (valeurs absurdes, forme inconnue, objet vide) : valider() remet
toute réponse dans les bornes, et VINYLE sert d'objet de repli si rien n'est exploitable.
"""

import re

from pochettes_ar.generation import ollama

MODELES_TEXTE = ["qwen3:8b", "qwen2.5:7b", "llama3.1:8b"]

FORMES = ("boite", "sphere", "cylindre", "cone")
ANIMATIONS = ("tourne", "flotte", "balance", "rebondit", "aucune")
AXES = ("aucun", "x", "y", "z")
MAX_PIECES = 16

# La partie qui change (description, couleurs) est à la fin : Ollama garde en mémoire le
# début d'une consigne déjà lue (cache des clés et valeurs de l'attention), et ne relit donc
# que la fin pour les pochettes suivantes (~1 s au lieu de ~18 s pour 2000 tokens).
CONSIGNE = """Tu construis un petit objet 3D de style jouet, très simplifié, qui représente
le sujet donné à la fin.

Procède en deux temps :
1. "plan" : liste en quelques mots les parties reconnaissables du sujet (3 à 12), comme le
   ferait un enfant avec des jouets en bois. Un personnage = corps, tête, bras, jambes ; un
   véhicule = caisse, cabine, roues ; un instrument = caisse, manche…
2. "pieces" : une forme simple par partie du plan, placée et dimensionnée pour que
   l'ensemble ressemble au sujet. Les pièces doivent se toucher, l'objet est posé sur le sol.

Formes : "boite" (pavé), "sphere" (sphère ou ellipsoïde), "cylindre" (axe vertical y),
"cone" (axe vertical, pointe vers le haut). Pour coucher un cylindre ou un cône, tourne-le
de 90° autour de x (il pointe alors vers l'avant, z) ou de z (il pointe alors vers x).
Repère : y vers le haut, sol en y = 0 ; x vers la droite ; z vers l'avant (vers le
spectateur). L'objet tient dans un cube de 10 unités, centré en x = 0, z = 0.
Pour chaque pièce : "position" = centre [x, y, z] ; "taille" = [largeur x, hauteur y,
profondeur z] (diamètres pour une sphère ou un cylindre) ; "rotation" = degrés [autour de x,
autour de y, autour de z] ; "couleur" = "#rrggbb" ; "brillance" de 0 (mat) à 1 (métal) ;
"tourne_sur_soi" = "x", "y" ou "z" pour une roue, une hélice ou un disque qui tourne, sinon "aucun".
"animation" de l'objet entier : "tourne", "flotte", "balance", "rebondit" ou "aucune".

Exemple 1, « une voiture rouge » :
{{"nom": "voiture", "plan": ["caisse", "cabine", "4 roues"], "animation": "rebondit", "pieces": [
 {{"forme": "boite", "position": [0, 2, 0], "taille": [8, 2, 4], "rotation": [0, 0, 0], "couleur": "#d42a2a", "brillance": 0.6, "tourne_sur_soi": "aucun"}},
 {{"forme": "boite", "position": [-0.5, 3.8, 0], "taille": [4, 1.8, 3.6], "rotation": [0, 0, 0], "couleur": "#9fd3ff", "brillance": 0.9, "tourne_sur_soi": "aucun"}},
 {{"forme": "cylindre", "position": [2.6, 1.1, 2.1], "taille": [2.2, 0.8, 2.2], "rotation": [90, 0, 0], "couleur": "#1a1a1a", "brillance": 0.1, "tourne_sur_soi": "z"}},
 {{"forme": "cylindre", "position": [-2.6, 1.1, 2.1], "taille": [2.2, 0.8, 2.2], "rotation": [90, 0, 0], "couleur": "#1a1a1a", "brillance": 0.1, "tourne_sur_soi": "z"}},
 {{"forme": "cylindre", "position": [2.6, 1.1, -2.1], "taille": [2.2, 0.8, 2.2], "rotation": [90, 0, 0], "couleur": "#1a1a1a", "brillance": 0.1, "tourne_sur_soi": "z"}},
 {{"forme": "cylindre", "position": [-2.6, 1.1, -2.1], "taille": [2.2, 0.8, 2.2], "rotation": [90, 0, 0], "couleur": "#1a1a1a", "brillance": 0.1, "tourne_sur_soi": "z"}}]}}

Exemple 2, « un musicien qui marche » :
{{"nom": "musicien", "plan": ["jambes", "corps", "bras", "tête", "cheveux"], "animation": "balance", "pieces": [
 {{"forme": "cylindre", "position": [-0.6, 1.8, 0.4], "taille": [0.9, 3.6, 0.9], "rotation": [-15, 0, 0], "couleur": "#222233", "brillance": 0.2, "tourne_sur_soi": "aucun"}},
 {{"forme": "cylindre", "position": [0.6, 1.8, -0.4], "taille": [0.9, 3.6, 0.9], "rotation": [15, 0, 0], "couleur": "#222233", "brillance": 0.2, "tourne_sur_soi": "aucun"}},
 {{"forme": "boite", "position": [0, 5, 0], "taille": [2.6, 3.2, 1.4], "rotation": [0, 0, 0], "couleur": "#f0f0f0", "brillance": 0.2, "tourne_sur_soi": "aucun"}},
 {{"forme": "cylindre", "position": [-1.6, 5, 0], "taille": [0.7, 3, 0.7], "rotation": [0, 0, -10], "couleur": "#f0f0f0", "brillance": 0.2, "tourne_sur_soi": "aucun"}},
 {{"forme": "cylindre", "position": [1.6, 5, 0], "taille": [0.7, 3, 0.7], "rotation": [0, 0, 10], "couleur": "#f0f0f0", "brillance": 0.2, "tourne_sur_soi": "aucun"}},
 {{"forme": "sphere", "position": [0, 7.4, 0], "taille": [1.6, 1.8, 1.6], "rotation": [0, 0, 0], "couleur": "#f1c6a5", "brillance": 0.2, "tourne_sur_soi": "aucun"}},
 {{"forme": "sphere", "position": [0, 7.9, -0.2], "taille": [1.8, 1.2, 1.7], "rotation": [0, 0, 0], "couleur": "#5a3a1a", "brillance": 0.1, "tourne_sur_soi": "aucun"}}]}}

Exemple 3, « un bonhomme de neige » :
{{"nom": "bonhomme de neige", "plan": ["boule du bas", "tête", "nez", "chapeau"], "animation": "flotte", "pieces": [
 {{"forme": "sphere", "position": [0, 2, 0], "taille": [4, 4, 4], "rotation": [0, 0, 0], "couleur": "#f4f6ff", "brillance": 0.1, "tourne_sur_soi": "aucun"}},
 {{"forme": "sphere", "position": [0, 5, 0], "taille": [3, 3, 3], "rotation": [0, 0, 0], "couleur": "#f4f6ff", "brillance": 0.1, "tourne_sur_soi": "aucun"}},
 {{"forme": "cone", "position": [0, 5.2, 1.8], "taille": [0.5, 1.2, 0.5], "rotation": [90, 0, 0], "couleur": "#ff8a1f", "brillance": 0.2, "tourne_sur_soi": "aucun"}},
 {{"forme": "cylindre", "position": [0, 7.2, 0], "taille": [2, 1.6, 2], "rotation": [0, 0, 0], "couleur": "#202020", "brillance": 0.4, "tourne_sur_soi": "aucun"}}]}}

Ne recopie pas les exemples. Construis maintenant : « {description} »
Couleurs de la pochette, à utiliser si elles conviennent : {couleurs}."""

_PIECE = {
    "type": "object",
    "properties": {
        "forme": {"type": "string", "enum": list(FORMES)},
        "position": {"type": "array", "items": {"type": "number"}},
        "taille": {"type": "array", "items": {"type": "number"}},
        "rotation": {"type": "array", "items": {"type": "number"}},
        "couleur": {"type": "string"},
        "brillance": {"type": "number"},
        "tourne_sur_soi": {"type": "string", "enum": list(AXES)},
    },
    "required": ["forme", "position", "taille", "rotation", "couleur", "brillance", "tourne_sur_soi"],
}
SCHEMA = {
    "type": "object",
    "properties": {
        "nom": {"type": "string"},
        "plan": {"type": "array", "items": {"type": "string"}},
        "animation": {"type": "string", "enum": list(ANIMATIONS)},
        "pieces": {"type": "array", "items": _PIECE},
    },
    "required": ["nom", "plan", "animation", "pieces"],
}

# Objet de repli : un disque vinyle qui tourne sur sa platine (si l'IA est absente ou échoue).
VINYLE = {
    "nom": "disque vinyle",
    "animation": "aucune",
    "pieces": [
        {"forme": "boite", "position": [0, 0.5, 0], "taille": [10, 1, 8.5], "rotation": [0, 0, 0],
         "couleur": "#3a2a1e", "brillance": 0.3, "tourne_sur_soi": "aucun"},
        {"forme": "cylindre", "position": [-0.8, 1.15, 0], "taille": [7, 0.3, 7], "rotation": [0, 0, 0],
         "couleur": "#111111", "brillance": 0.8, "tourne_sur_soi": "y"},
        {"forme": "cylindre", "position": [-0.8, 1.35, 0], "taille": [2.4, 0.12, 2.4], "rotation": [0, 0, 0],
         "couleur": "#d8442c", "brillance": 0.2, "tourne_sur_soi": "y"},
        {"forme": "cylindre", "position": [3.8, 2.2, -3], "taille": [0.5, 2.4, 0.5], "rotation": [0, 0, 0],
         "couleur": "#c0c0c8", "brillance": 0.9, "tourne_sur_soi": "aucun"},
        {"forme": "boite", "position": [2.6, 3.3, -1.2], "taille": [0.3, 0.3, 4.4], "rotation": [0, -35, 0],
         "couleur": "#c0c0c8", "brillance": 0.9, "tourne_sur_soi": "aucun"},
    ],
}


def _nombres(valeurs, n, defaut, mini=-1e3, maxi=1e3):
    """Liste de n nombres finis dans [mini, maxi], complétée par défaut si la réponse est courte."""
    sortie = []
    for k in range(n):
        try:
            x = float(valeurs[k])
            x = x if x == x and abs(x) != float("inf") else defaut[k]  # NaN, infini -> défaut
        except (IndexError, TypeError, ValueError):
            x = defaut[k]
        sortie.append(min(max(x, mini), maxi))
    return sortie


def couleur_rgb(texte, defaut=(0.8, 0.8, 0.8)):
    """« #rrggbb » (ou « rrggbb », « #rgb ») -> (r, g, b) dans [0, 1]."""
    m = re.fullmatch(r"#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})", str(texte).strip())
    if not m:
        return tuple(defaut)
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def valider(composition):
    """Remet une composition dans les bornes ; lève ValueError si aucune pièce n'est utilisable."""
    if not isinstance(composition, dict):
        raise ValueError("la composition n'est pas un objet JSON")
    pieces = []
    for p in composition.get("pieces", [])[:MAX_PIECES]:
        if not isinstance(p, dict):
            continue
        forme = p.get("forme") if p.get("forme") in FORMES else "boite"
        taille = _nombres(p.get("taille", []), 3, [1, 1, 1], mini=0.05, maxi=20)
        pieces.append({
            "forme": forme,
            "position": _nombres(p.get("position", []), 3, [0, 0, 0], mini=-20, maxi=20),
            "taille": taille,
            "rotation": _nombres(p.get("rotation", []), 3, [0, 0, 0], mini=-360, maxi=360),
            "couleur": p.get("couleur") if isinstance(p.get("couleur"), str) else "#cccccc",
            "brillance": _nombres([p.get("brillance")], 1, [0.3], mini=0, maxi=1)[0],
            "tourne_sur_soi": p.get("tourne_sur_soi") if p.get("tourne_sur_soi") in AXES else "aucun",
        })
    if not pieces:
        raise ValueError("aucune pièce utilisable")
    animation = composition.get("animation")
    plan = composition.get("plan", [])
    return {
        "nom": str(composition.get("nom", "objet"))[:60],
        "plan": [str(x)[:60] for x in plan][:MAX_PIECES] if isinstance(plan, list) else [],
        "animation": animation if animation in ANIMATIONS else "tourne",
        "pieces": pieces,
    }


def composer(description, couleurs=(), modele=None):
    """Composition validée de l'objet décrit. Lève OllamaIndisponible ou ValueError en cas d'échec."""
    modele = modele or ollama.choisir(MODELES_TEXTE)
    if modele is None:
        raise ollama.OllamaIndisponible(f"aucun modèle de texte installé parmi {MODELES_TEXTE}")
    consigne = CONSIGNE.format(description=description, couleurs=", ".join(couleurs) or "libres")
    composition = valider(ollama.discuter(modele, consigne, schema=SCHEMA, temperature=0.4))
    composition["modele"] = modele
    return composition
