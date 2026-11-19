"""Capture webcam (cv2.VideoCapture)."""

import sys

import cv2


def ouvrir_camera(indice=0, largeur=1280, hauteur=720):
    """Ouvre la webcam et demande une résolution (la caméra peut en choisir une autre)."""
    camera = cv2.VideoCapture(indice, cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    if not camera.isOpened():
        raise RuntimeError(
            f"Impossible d'ouvrir la caméra {indice}. Sous macOS, autoriser l'accès à la caméra pour "
            "le terminal (Réglages Système > Confidentialité et sécurité > Caméra)."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, largeur)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, hauteur)
    return camera
