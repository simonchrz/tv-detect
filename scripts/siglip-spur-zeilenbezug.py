#!/usr/bin/env python3
"""Welche SigLIP-Variante passt zu den ZEILEN des Kopfs? (Nachtrag zur
Zeitachsen-Pruefung, siglip-spur-design.md Schritt 1.)

Der Kopf sieht die Backbone-Merkmale aus tv-detect (Archiv/Feature-Cache).
Szenenschnitte erscheinen in beiden Merkmalsarten als Sprung von einer
Zeile zur naechsten. Die Verschiebung, bei der die Sprung-Reihen am besten
korrelieren, zeigt, wie eine SigLIP-Variante zu den Kopf-Zeilen liegt —
unabhaengig von beiden Modellen und von der absoluten Zeit.
Varianten: am_stueck (O28-Cache) und gekachelt (siglip-spur.py, neu gerechnet).
Laeuft in ~/ml/siglip-exp/.venv. Schreibt nichts in Caches.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
C = Path.home() / ".cache"
spec = importlib.util.spec_from_file_location("siglip_spur", HIER / "siglip-spur.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)


def spruenge_bb(F):
    F = np.asarray(F, np.float32)
    return np.linalg.norm(np.diff(F, axis=0), axis=1)


def spruenge_sig(E):
    E = np.asarray(E, np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    return 1 - (E[1:] * E[:-1]).sum(1)


def bester_lag(a, b, lags=range(-3, 4)):
    """Korrelation von a[t] mit b[t+lag]; b = SigLIP. Liefert (lag, {lag: r})."""
    r = {}
    for L in lags:
        if L >= 0:
            x, y = a[:len(a) - L], b[L:]
        else:
            x, y = a[-L:], b[:len(b) + L]
        n = min(len(x), len(y))
        r[L] = float(np.corrcoef(x[:n], y[:n])[0, 1])
    return max(r, key=r.get), r


def main():
    uu = [z["uuid"] for z in json.load(open(C / "tvd-siglip-zeitachse.json"))]
    enc = sp.Encoder()
    for u in uu:
        meta = json.loads(str(np.load(C / "tvd-train-archive" / f"{u}.npz", allow_pickle=True)["meta"]))
        F = np.load(meta["feature_npy"], mmap_mode="r")[:, :1280]
        A = np.load(C / "tvd-siglip2" / f"{u}.npy")
        K, _, _ = sp.spur(C / "tv-detect-daemon/source" / f"{u}.ts", enc)
        d = spruenge_bb(F)
        la, ra = bester_lag(d, spruenge_sig(A))
        lk, rk = bester_lag(d, spruenge_sig(K))
        # zweite Haelfte getrennt: Drift zeigt sich als anderer Lag hinten
        h = len(d) // 2
        la2, _ = bester_lag(d[h:], spruenge_sig(A)[h:])
        lk2, _ = bester_lag(d[h:], spruenge_sig(K)[h:])
        print(json.dumps({"uuid": u, "kopf_zeilen": len(F), "am_stueck": len(A), "gekachelt": len(K),
                          "lag_alt": la, "r_alt": round(ra[la], 3), "r_alt0": round(ra[0], 3), "lag_alt_hinten": la2,
                          "lag_neu": lk, "r_neu": round(rk[lk], 3), "r_neu0": round(rk[0], 3), "lag_neu_hinten": lk2}),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
