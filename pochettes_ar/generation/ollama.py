"""Client minimal de l'API HTTP d'Ollama, le serveur de modèles d'IA qui tourne en local.

Ollama (https://ollama.com) fait tourner des modèles de langage et de vision sur le Mac (puce
Apple, sans carte NVIDIA) et les expose sur http://localhost:11434. On l'appelle avec la
bibliothèque standard (urllib), sans dépendance supplémentaire.

Sorties structurées : on passe un schéma JSON dans le champ « format », et Ollama contraint
le modèle à produire un JSON qui le respecte. C'est ce qui rend les réponses exploitables
directement par le code (description, composition de l'objet 3D).

Installation (déjà faite sur ce Mac) : brew install ollama && brew services start ollama
Modèles : ollama pull qwen3-vl:8b ; ollama pull qwen3:8b
"""

import base64
import json
import urllib.error
import urllib.request

ADRESSE = "http://localhost:11434"


class OllamaIndisponible(RuntimeError):
    """Le serveur Ollama ne répond pas, ou aucun des modèles voulus n'est installé."""


def _requete(chemin, donnees=None, timeout=5.0):
    corps = None if donnees is None else json.dumps(donnees).encode()
    req = urllib.request.Request(ADRESSE + chemin, data=corps, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as reponse:
            return json.loads(reponse.read())
    except urllib.error.HTTPError as erreur:
        raise OllamaIndisponible(f"{chemin} : HTTP {erreur.code} {erreur.read().decode(errors='replace')[:300]}")
    except (urllib.error.URLError, TimeoutError, ConnectionError) as erreur:
        raise OllamaIndisponible(f"Ollama ne répond pas sur {ADRESSE} ({erreur})")


def modeles_installes():
    """Noms des modèles installés (ex. ["qwen3:8b", "qwen3-vl:8b"]) ; [] si Ollama ne répond pas."""
    try:
        return [m["name"] for m in _requete("/api/tags", timeout=2.0).get("models", [])]
    except OllamaIndisponible:
        return []


def choisir(preferences, installes=None):
    """Premier modèle de la liste de préférences qui est installé, ou None."""
    installes = modeles_installes() if installes is None else installes
    for nom in preferences:
        if nom in installes or f"{nom}:latest" in installes:
            return nom
    return None


def encoder_image(octets_jpeg):
    return base64.b64encode(octets_jpeg).decode("ascii")


def discuter(modele, consigne, images=(), schema=None, temperature=0.2, timeout=300.0, contexte=8192):
    """Envoie une consigne (et éventuellement des images JPEG) au modèle ; renvoie sa réponse.

    Avec un schéma JSON, la réponse est décodée et renvoyée sous forme de dict."""
    message = {"role": "user", "content": consigne}
    if images:
        message["images"] = [encoder_image(i) for i in images]
    donnees = {
        "model": modele,
        "messages": [message],
        "stream": False,
        # contexte : nombre de tokens (consigne + image + réponse) ; 4096 par défaut, trop juste
        # pour la consigne de composition avec ses exemples et une réponse de 12 pièces.
        "options": {"temperature": temperature, "num_ctx": contexte},
        # Les modèles « qui réfléchissent » (Qwen 3) écriraient d'abord un long raisonnement :
        # inutile ici, et beaucoup plus lent.
        "think": False,
    }
    if schema is not None:
        donnees["format"] = schema
    try:
        reponse = _requete("/api/chat", donnees, timeout)
    except OllamaIndisponible as erreur:
        if "think" not in str(erreur):
            raise
        del donnees["think"]  # modèle qui ne connaît pas l'option : on réessaie sans
        reponse = _requete("/api/chat", donnees, timeout)
    texte = reponse["message"]["content"]
    return json.loads(texte) if schema is not None else texte
