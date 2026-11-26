# Projet pochettes AR

Projet de cours d'informatique graphique (référence : Marschner & Shirley, *Fundamentals of Computer Graphics* — viewing, graphics pipeline, shading, texture mapping).

Une webcam reconnaît une pochette d'album et affiche par-dessus, en réalité augmentée, un objet 3D animé propre à chaque album.
But : n'importe quelle pochette. Les 5 albums de `albums/` ont des modèles faits à la main ; une pochette inconnue est découverte par la webcam, décrite en quelques mots par une IA de vision locale, puis une IA de texte compose un objet très simple en primitives (JSON) que notre code construit.

## Contraintes (non négociables)

- **Langage : Python uniquement.**
- **Vision : OpenCV autorisé** — capture webcam, détection/description ORB, appariement, homographie/RANSAC, `solvePnP`, calibration, affichage de la frame finale (`cv2.imshow`).
- **Rendu 3D : écrit par nous, c'est le cœur du cours.** Projection, matrices de vue, rasterisation, z-buffer, shading, placage de textures sont implémentés à la main avec NumPy (et Numba pour accélérer *notre* code).
  - Interdit : OpenGL / PyOpenGL, moderngl, pyrender, Open3D, trimesh pour le rendu, pygame 3D, Panda3D, Blender, ou toute bibliothèque qui rasterise à notre place.
  - Interdit aussi pour la partie 3D : `cv2.projectPoints`, `cv2.fillPoly`, `cv2.drawContours`, `cv2.warpPerspective`, etc. La projection et le remplissage des triangles se font dans `pochettes_ar/rendu/`. (OpenCV peut servir au débogage de la vision, pas pour produire l'image 3D.)
  - Le chargeur de modèles (OBJ) est aussi écrit à la main (`pochettes_ar/scene/maillage.py`).
  - IA pour les pochettes inconnues : **Ollama en local** (Mac M4, pas de cloud, pas de clé), appelé en HTTP avec la bibliothèque standard (`generation/ollama.py`) ; vision `qwen3-vl:8b-instruct` (surtout pas la variante « thinking » : raisonnement caché, 40 s de plus), texte `qwen3:8b`. L'IA ne produit que du JSON (description, puis pièces : boîte/sphère/cylindre/cône + position/taille/rotation/couleur) ; la géométrie est construite par notre code (`scene/primitives.py`). Ne pas ajouter de SDK d'IA aux dépendances.
  - Ancienne piste, gardée pour les 5 albums : rembg + TripoSR + simplification (`fast-simplification`, `trimesh`) tournent **hors du projet**, sur Google Colab (`outils/generer_modeles_ia.ipynb` ; pas de GPU NVIDIA ici : Mac M4). Ils ne produisent que des .obj dans `cache/` ; le chargement et le rendu restent les nôtres. Ne pas ajouter torch/rembg/trimesh aux dépendances du projet.
  - Tolérés hors du pipeline AR : `matplotlib` uniquement dans le visualiseur de contrôle `outils/visualiser.py` (provisoire, en attendant notre moteur) ; le dessin 2D OpenCV (`cv2.putText`, `cv2.circle`…) uniquement pour fabriquer les images de texture dans `outils/generer_modeles.py`.
- Le code doit rester lisible et pédagogique : rattacher les étapes du pipeline aux chapitres du livre dans les docstrings.

## Organisation

```
pochettes_ar/
  vision/        # OpenCV : capture, reconnaissance (ORB + ratio test + vote + RANSAC ; ajouter_reference en cours de route), affichage 2D de contrôle, pose (solvePnP), calibration,
                 #   decouverte (quadrilatère inconnu : Canny par canal couleur + Douglas-Peucker + contrôle d'appui sur les bords, stable ~0.7 s ; ou cadre guide + touche n ; redressement par homographie),
                 #   lissage (pose lissée par album : quaternions + moyenne exponentielle adaptative, rejet des bascules IPPE)
  rendu/         # maison : transformations, camera (P depuis K + viewport), rasterisation (Numba), zbuffer,
                 #   textures (bilinéaire), shading (Blinn-Phong), compositing, objet (buffers), pipeline (orchestration)
  scene/         # maillages + chargeur OBJ, primitives (sphère, pavé, cylindre, cône…), animations par modèle (t -> état animé), catalogue album -> modèle
  generation/    # pochettes inconnues : ollama (client HTTP), description (vision : sujet, texte lu, album reconnu),
                 #   identification (nom de l'album : texte lu confirmé par MusicBrainz, ou album reconnu par l'IA vérifié par MusicBrainz
                 #   + comparaison ORB/homographie avec la pochette officielle du Cover Art Archive ; sinon pas de nom plutôt qu'un faux),
                 #   musicbrainz (requêtes espacées de 1,1 s : au-delà, 503), composition (texte -> JSON validé, repli VINYLE),
                 #   construction (JSON -> maillage, normalisé sur la pochette), generateur (fil d'exécution à part + cache/pochettes/<id>/)
  main.py        # boucle capture -> reconnaissance -> pose -> rendu -> incrustation -> affichage
cache/pochettes/ # pochettes découvertes : reference.jpg (image redressée = référence ORB) + pochette.json (titre, description, composition) ; rechargées au lancement, sans IA
cache/           # modèles générés par IA (<id_album>.obj/.mtl, couleur par sommet), décompressés depuis cache.zip (Colab) ; prioritaires sur modeles/
albums/          # images de référence des pochettes + albums.json (id -> fichier, MBID MusicBrainz)
outils/          # scripts annexes : telecharger_pochettes.py, generer_modeles.py, visualiser.py, generer_modeles_ia.ipynb (Colab)
modeles/         # modèles 3D générés (.obj/.mtl/.png) + modeles.json (album -> modèle, pivots, animation)
calibration/     # paramètres intrinsèques de la caméra (K, distorsion)
tests/           # pytest : modèles, reconnaissance sur scènes synthétiques, (à venir) rendu
```

Pont vision ↔ rendu : `solvePnP` donne `rvec, tvec` (pose de la pochette dans le repère caméra) et la calibration donne `K` ; c'est notre code de `rendu/camera.py` qui en construit les matrices de vue et de projection.
Sans `calibration/camera.json`, `K` est approximée (champ horizontal de 60°, centre de l'image, sans distorsion).

## Pipeline de rendu (`pochettes_ar/rendu/pipeline.py`)

- `Pipeline(largeur, hauteur, K).rendre(image_webcam, [Appel(objet, pose.matrice(), **animer(entree, objet, t))])`.
- État animé (`scene/animations.py`, une fonction par modèle, pivots lus dans modeles.json) : `matrices_parties` (rigide, par partie), `decalages_uv` (texture qui défile, par partie), `sommets`/`normales` (déformation, ex. billet qui ondule).
- Sommets : modèle -> pochette (matrice par partie, pour l'animation) -> caméra (pose) -> clip (P construite depuis K) -> /w -> viewport. Convention OpenCV gardée jusqu'aux pixels (y vers le bas, centre du pixel (0, 0) en 0).
- Rejets : triangles avec un sommet derrière le plan proche (pas de vrai clipping), faces arrière (aire signée >= 0 à l'écran), hors image.
- Rasterisation (seule boucle par pixel, compilée par Numba) : barycentriques par fonctions d'arête, z-buffer sur z_ndc, barycentriques corrigées de la perspective stockées par pixel.
- Shading différé en NumPy : interpolation position/normale/UV du triangle visible, texture bilinéaire, Blinn-Phong dans le repère caméra ; lumière directionnelle fixée à la pochette. `ombrage` "phong" ou "plat" par modèle (modeles.json).
- Pas d'anticrénelage, pas d'ombres portées.

## Modèles 3D

- Générés par le code (`outils/generer_modeles.py`) : ne pas éditer les .obj à la main, modifier le script et relancer.
- Repère commun : unité = largeur de la pochette, origine au centre ; pochette = carré [-0.5, 0.5]² du plan y = 0 ; Y = normale sortante ; +X = droite, +Z = bord bas de la pochette (côté spectateur) ; les objets regardent vers +Z. `solvePnP` reçoit les coins dans ce même repère (`vision/pose.py`) : `Pose.matrice()` est directement la matrice modèle -> caméra (convention caméra OpenCV : x droite, y bas, z vers l'avant).
- Triangles dans le sens trigonométrique vus de l'extérieur, indices unifiés (v/vt/vn identiques), UV avec v = 0 en bas de l'image.
- Un objet `o` par partie animable ; pivots et axes dans `modeles/modeles.json`.
- Modèles IA (`cache/`) : une seule partie `ia`, ~3000 triangles, `v x y z r g b` (couleur par sommet, interpolée puis multipliée par Kd), faces sans normales (normales lissées calculées au chargement), même repère. `Catalogue` les préfère s'ils existent et se lisent, sinon repli sur `modeles/` ; animation générique `ia` (rotation + flottement). Touche `g` : IA / classiques.

## Environnement

- Géré avec `uv` (Python 3.14, `.venv/` dans le projet). Dépendances : `numpy`, `opencv-python`, `numba` ; dev : `pytest`, `matplotlib` (visualiseur seulement).
- Lancer : `uv run python -m pochettes_ar.main` (`--camera N`, ou `--image photo.jpg`) : reconnaissance, pose lissée, rendu du modèle animé ; touches i/a/c/m/l/p/g/d/n (infos, axes, contour, modèle, lissage, pause, modèles IA ou classiques, découverte automatique, capture du cadre guide)
- Tests : `uv run pytest` (sans IA) ; `uv run pytest -m lent` : test avec la vraie IA (~1 min)
- IA locale : `brew services start ollama` ; modèles : `ollama pull qwen3-vl:8b-instruct` et `ollama pull qwen3:8b`. Sans Ollama, une pochette inconnue reçoit l'objet de repli (platine vinyle). Une génération prend ~50 s (en arrière-plan) ; la boucle webcam reste à ~20 ms par image.
- Régénérer les modèles : `uv run python -m outils.generer_modeles`
- Les vérifier : `uv run python -m outils.visualiser [casque ...] [--serre] [--normales] [--aretes]`
- Ajouter une dépendance : `uv add <paquet>` — vérifier d'abord qu'elle ne fait pas le rendu 3D à notre place.
