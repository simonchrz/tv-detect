"""SigLIP-Spalten — EINE Definition fuer Training, Messung und die Go-Paritaet
(Muster: ocr_spalten.py). Entwurf docs/siglip-spur-design.md.

Spur = rohe SigLIP-2-Merkmale je Sekunde (scripts/siglip-spur.py), Zeile i =
Sekunde i. Spalten = 64 Hauptkomponenten + siglip_da. Mittelwert mu und
Projektion V stehen im MLP7-Kopf (write_mlp_head_v7); Go rechnet dieselbe
Formel in internal/signals/siglipspalten.go.

Die Projektion ist WEISSEND (Komponente / Wurzel des Eigenwerts): der
Produktionskopf standardisiert seine Eingaben nicht, und rohe PCA-Werte
laegen um Groessenordnungen neben den uebrigen Spalten.
"""
from pathlib import Path

import numpy as np

DIM = 768
K = 64
N_SPALTEN = K + 1


def projektion_anpassen(zeilen, k=K):
    """mu (768,), V (768, k) float32 aus train-Zeilen MIT Spur."""
    Z = np.asarray(zeilen, np.float32)
    mu = Z.mean(0)
    w, U = np.linalg.eigh(np.cov((Z - mu).T))
    ordnung = np.argsort(w)[::-1][:k]
    V = U[:, ordnung] / np.sqrt(np.maximum(w[ordnung], 1e-12))
    # Vorzeichen festlegen: eigh liefert jede Komponente mit beliebigem
    # Vorzeichen. Zwischen 02.10. und 03.10.2026 drehten 28 von 64, und das
    # Gate scorte den Champion auf halb gespiegelten Spalten. Konvention: der
    # betragsgroesste Eintrag jeder Spalte ist positiv.
    groesste = np.abs(V).argmax(0)
    V = V * np.where(V[groesste, np.arange(V.shape[1])] < 0, -1.0, 1.0)
    return mu.astype(np.float32), V.astype(np.float32)


def spalten(E, n_sek, mu, V):
    """(n_sek, 65) float32. Sekunden ausserhalb der Spur (oder E=None): 0."""
    aus = np.zeros((n_sek, N_SPALTEN), np.float32)
    if E is None:
        return aus
    m = min(n_sek, len(E))
    aus[:m, :K] = (np.asarray(E[:m], np.float32) - mu) @ V
    aus[:m, K] = 1.0
    return aus


def lade_spur(uuid, spur_dir):
    """Rohe Spur oder None. Nur wenn die Frische-Beilage (.json) existiert —
    ohne sie gilt eine Spur als nicht vorhanden (siglip-spur.py schreibt die
    JSON zuletzt)."""
    d = Path(spur_dir)
    if not (d / f"{uuid}.npy").is_file() or not (d / f"{uuid}.json").is_file():
        return None
    return np.load(d / f"{uuid}.npy", mmap_mode="r")
