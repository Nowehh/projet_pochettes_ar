"""Structure de maillage (sommets, normales, UV, faces) et chargeur OBJ écrit à la main.

Conventions :
- triangles orientés dans le sens trigonométrique vus de l'extérieur (la normale géométrique
  (b - a) x (c - a) pointe du même côté que les normales aux sommets) ;
- indices « unifiés » : le sommet i a une position, une normale et des UV (un sommet OBJ
  v/vt/vn différent devient un sommet distinct) ;
- UV dans [0, 1], v = 0 en bas de l'image (convention OBJ).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Materiau:
    nom: str
    kd: tuple = (0.8, 0.8, 0.8)  # couleur diffuse
    ks: tuple = (0.0, 0.0, 0.0)  # couleur spéculaire
    ns: float = 10.0  # exposant de Phong
    texture: str | None = None  # chemin relatif au fichier .obj


@dataclass
class Maillage:
    sommets: np.ndarray  # (N, 3)
    normales: np.ndarray  # (N, 3), unitaires
    uv: np.ndarray  # (N, 2)
    triangles: np.ndarray  # (M, 3), indices dans les tableaux ci-dessus

    def __post_init__(self):
        self.sommets = np.asarray(self.sommets, dtype=np.float64).reshape(-1, 3)
        self.normales = np.asarray(self.normales, dtype=np.float64).reshape(-1, 3)
        self.uv = np.asarray(self.uv, dtype=np.float64).reshape(-1, 2)
        self.triangles = np.asarray(self.triangles, dtype=np.int64).reshape(-1, 3)

    @staticmethod
    def fusion(maillages):
        """Concatène plusieurs maillages en décalant les indices des triangles."""
        decalages = np.cumsum([0] + [len(m.sommets) for m in maillages[:-1]])
        return Maillage(
            np.concatenate([m.sommets for m in maillages]),
            np.concatenate([m.normales for m in maillages]),
            np.concatenate([m.uv for m in maillages]),
            np.concatenate([m.triangles + d for m, d in zip(maillages, decalages)]),
        )


@dataclass
class Partie:
    """Un objet nommé du fichier OBJ (ex. une roue), avec un seul matériau."""

    nom: str
    maillage: Maillage
    materiau: Materiau


def calculer_normales(sommets, triangles):
    """Normales lissées : moyenne des normales des faces adjacentes, pondérée par leur aire."""
    a, b, c = (sommets[triangles[:, k]] for k in range(3))
    faces = np.cross(b - a, c - a)  # norme = 2 x aire
    normales = np.zeros_like(sommets)
    for k in range(3):
        np.add.at(normales, triangles[:, k], faces)
    longueurs = np.linalg.norm(normales, axis=1, keepdims=True)
    return normales / np.where(longueurs > 0, longueurs, 1.0)


# --- Écriture -------------------------------------------------------------------------


def _vec(valeurs):
    return " ".join(f"{x:.6g}" for x in valeurs)


def ecrire_obj(chemin, parties):
    """Écrit parties -> chemin (.obj) et le .mtl associé, à côté."""
    chemin = Path(chemin)
    chemin_mtl = chemin.with_suffix(".mtl")

    materiaux = {p.materiau.nom: p.materiau for p in parties}
    mtl = []
    for m in materiaux.values():
        mtl += [f"newmtl {m.nom}", f"Kd {_vec(m.kd)}", f"Ks {_vec(m.ks)}", f"Ns {m.ns:.6g}"]
        if m.texture:
            mtl.append(f"map_Kd {m.texture}")
        mtl.append("")
    chemin_mtl.write_text("\n".join(mtl))

    lignes = [f"# {chemin.name} — généré par outils/generer_modeles.py", f"mtllib {chemin_mtl.name}"]
    decalage = 1  # les indices OBJ commencent à 1
    for p in parties:
        m = p.maillage
        lignes += [f"o {p.nom}", f"usemtl {p.materiau.nom}"]
        lignes += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in m.sommets]
        lignes += [f"vt {u:.6f} {v:.6f}" for u, v in m.uv]
        lignes += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in m.normales]
        for tri in m.triangles + decalage:
            lignes.append("f " + " ".join(f"{i}/{i}/{i}" for i in tri))
        decalage += len(m.sommets)
    chemin.write_text("\n".join(lignes) + "\n")


# --- Lecture --------------------------------------------------------------------------


def _charger_mtl(chemin):
    materiaux, courant = {}, None
    for ligne in Path(chemin).read_text().splitlines():
        mots = ligne.split()
        if not mots or mots[0].startswith("#"):
            continue
        if mots[0] == "newmtl":
            courant = materiaux[mots[1]] = Materiau(mots[1])
        elif courant is None:
            continue
        elif mots[0] == "Kd":
            courant.kd = tuple(float(x) for x in mots[1:4])
        elif mots[0] == "Ks":
            courant.ks = tuple(float(x) for x in mots[1:4])
        elif mots[0] == "Ns":
            courant.ns = float(mots[1])
        elif mots[0] == "map_Kd":
            courant.texture = mots[-1]
    return materiaux


def _indice(texte, n):
    """Indice OBJ (1-based, ou négatif = relatif à la fin) -> indice Python, ou None."""
    if not texte:
        return None
    i = int(texte)
    return i - 1 if i > 0 else n + i


def charger_obj(chemin):
    """Lit un .obj (et son .mtl) -> liste de Partie. Polygones triangulés en éventail."""
    chemin = Path(chemin)
    positions, coords_uv, normales = [], [], []
    materiaux = {}
    groupes = []  # [nom, nom_materiau, liste de faces (triplets v/vt/vn)]
    nom, nom_mat = "defaut", None

    def groupe_courant():
        if not groupes or groupes[-1][0] != nom or groupes[-1][1] != nom_mat:
            groupes.append([nom, nom_mat, []])
        return groupes[-1][2]

    for ligne in chemin.read_text().splitlines():
        mots = ligne.split()
        if not mots or mots[0].startswith("#"):
            continue
        cle, args = mots[0], mots[1:]
        if cle == "v":
            positions.append([float(x) for x in args[:3]])
        elif cle == "vt":
            coords_uv.append([float(x) for x in args[:2]])
        elif cle == "vn":
            normales.append([float(x) for x in args[:3]])
        elif cle in ("o", "g"):
            nom = args[0] if args else "defaut"
        elif cle == "usemtl":
            nom_mat = args[0]
        elif cle == "mtllib":
            chemin_mtl = chemin.parent / " ".join(args)
            if chemin_mtl.exists():
                materiaux.update(_charger_mtl(chemin_mtl))
        elif cle == "f":
            coins = []
            for coin in args:
                champs = (coin.split("/") + ["", ""])[:3]
                coins.append(
                    (
                        _indice(champs[0], len(positions)),
                        _indice(champs[1], len(coords_uv)),
                        _indice(champs[2], len(normales)),
                    )
                )
            faces = groupe_courant()
            for k in range(1, len(coins) - 1):
                faces.append((coins[0], coins[k], coins[k + 1]))

    parties, noms_vus = [], {}
    for nom_g, nom_mat_g, faces in groupes:
        if not faces:
            continue
        unifies, triangles = {}, []
        for face in faces:
            triangles.append([unifies.setdefault(coin, len(unifies)) for coin in face])
        coins = list(unifies)
        sommets = np.array([positions[v] for v, _, _ in coins])
        uv = np.array([coords_uv[t] if t is not None else (0.0, 0.0) for _, t, _ in coins])
        triangles = np.array(triangles)
        if all(n is not None for _, _, n in coins):
            norm = np.array([normales[n] for _, _, n in coins], dtype=np.float64)
            norm /= np.linalg.norm(norm, axis=1, keepdims=True)
        else:
            norm = calculer_normales(sommets, triangles)

        # Un même nom d'objet avec plusieurs matériaux donne plusieurs parties.
        noms_vus[nom_g] = noms_vus.get(nom_g, 0) + 1
        nom_partie = nom_g if noms_vus[nom_g] == 1 else f"{nom_g}.{noms_vus[nom_g]}"
        materiau = materiaux.get(nom_mat_g) or Materiau(nom_mat_g or "defaut")
        parties.append(Partie(nom_partie, Maillage(sommets, norm, uv, triangles), materiau))
    return parties
