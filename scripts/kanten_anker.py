#!/usr/bin/env python3
"""Anker-Verfügbarkeit an den Kantenfehlern: trägt ein Material-Snap?

Für jede verschobene Kante (aus der Fehlermoden-Zerlegung) wird gemessen:

  1. Liegt an der WAHREN Kante (menschliches Label) ein Logo-Wechsel in
     Reichweite? → Ob ein Snap die richtige Stelle überhaupt finden KANN.
  2. Liegt die AUTO-Kante bereits auf einem Anker? → Ob der Fehler ein
     „falscher Anker" ist statt „kein Anker" — dann hilft Snappen nichts,
     dann konkurrieren zwei Anker um die Semantik.
  3. Naiver Snap simuliert: Auto-Kante auf den nächsten Logo-Wechsel
     (±10 s) gezogen — wird der Fehler kleiner oder größer?

Anker v1 = Logo-Wechsel (|Δlogo| > 0.3 zwischen Nachbarsekunden) aus der
Feature-Spalte 1280 der gecachten .npy (Semantik verifiziert: logo<0.5 zu
86 % im Werbeblock, 0,4 % außerhalb). Tonpegel-Dips und Schwarzbilder sind
bewusst NICHT drin — erst das billigste Signal ausschöpfen, dann erweitern.

⚠️ Stichproben-Befund, der die Messung motiviert (kabel-eins 1778511363):
die Auto-Kante saß AUF einem Logo-Wechsel (1230), die menschliche 20 s
später OHNE Anker (Trailer-Zone, Logo bleibt an). Anker können der
Label-Semantik widersprechen — deshalb wird beides gezählt.
"""
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
ARCHIV = Path.home() / ".cache/tvd-train-archive"
FEATURES = Path.home() / ".cache/tvd-features"

spec = importlib.util.spec_from_file_location("fm", REPO / "fehlermoden.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)


def logo_wechsel(uuid):
    """Sekunden-Indizes der Logo-Übergänge, oder None ohne Feature-Datei."""
    treffer = sorted(glob.glob(str(FEATURES / f"{uuid}-*fps100-l2-a1.npy")))
    if not treffer:
        return None
    a = np.load(treffer[-1], mmap_mode="r")
    logo = np.asarray(a[:, 1280], dtype=np.float32)
    logo = np.nan_to_num(logo, nan=0.5)
    d = np.abs(np.diff(logo))
    return np.where(d > 0.3)[0].astype(np.float64) + 0.5  # Wechsel ~Sekundenmitte


def naechster(anker, t, fenster):
    if anker is None or len(anker) == 0:
        return None
    d = np.abs(anker - t)
    i = int(np.argmin(d))
    return float(anker[i]) if d[i] <= fenster else None


def main():
    uuids = json.loads((ARCHIV / "golden-eval-set.json").read_text())["uuids"]
    kanten = []          # (uuid, wahr, auto)
    ohne_features = 0
    for u in uuids:
        d = fm.SNAPSHOT / f"_rec_{u}"
        user, auto = fm.bloecke(d / "ads_user.json"), fm.bloecke(d / "ads.json")
        if user is None or auto is None:
            continue
        z = fm.zerlege(auto, user)
        for ub, ab, ds, de in z["grenze"]:
            if abs(ds) > 0.5:
                kanten.append((u, ub[0], ab[0]))
            if abs(de) > 0.5:
                kanten.append((u, ub[1], ab[1]))

    W_TRUE, W_SNAP = 5.0, 10.0
    hat_anker = kein_anker = auto_auf_anker = 0
    besser = schlechter = gleich = 0
    fehler_alt, fehler_neu = [], []
    anker_cache = {}
    for u, wahr, autok in kanten:
        if u not in anker_cache:
            anker_cache[u] = logo_wechsel(u)
        anker = anker_cache[u]
        if anker is None:
            ohne_features += 1
            continue
        a_true = naechster(anker, wahr, W_TRUE)
        if a_true is not None:
            hat_anker += 1
        else:
            kein_anker += 1
        if naechster(anker, autok, 1.5) is not None:
            auto_auf_anker += 1
        # naiver Snap: Auto-Kante auf naechsten Anker im ±10-s-Fenster
        ziel = naechster(anker, autok, W_SNAP)
        alt = abs(autok - wahr)
        neu = abs(ziel - wahr) if ziel is not None else alt
        fehler_alt.append(alt)
        fehler_neu.append(neu)
        if neu < alt - 0.5:
            besser += 1
        elif neu > alt + 0.5:
            schlechter += 1
        else:
            gleich += 1

    n = len(fehler_alt)
    print(f"{len(kanten)} verschobene Kanten, {n} mit Features "
          f"({ohne_features} ohne)\n")
    print(f"1. Anker an der WAHREN Kante (±{W_TRUE:.0f}s): "
          f"{hat_anker}/{n} ({100*hat_anker//max(n,1)} %) — "
          f"{kein_anker} wahre Kanten OHNE Logo-Anker")
    print(f"2. Auto-Kante sitzt bereits auf einem Anker (±1.5s): "
          f"{auto_auf_anker}/{n} ({100*auto_auf_anker//max(n,1)} %)")
    print(f"3. Naiver Logo-Snap (±{W_SNAP:.0f}s): "
          f"{besser} besser, {gleich} unveraendert, {schlechter} SCHLECHTER")
    print(f"   Kantenfehler Median: {np.median(fehler_alt):.1f}s → "
          f"{np.median(fehler_neu):.1f}s   "
          f"Summe: {sum(fehler_alt):.0f}s → {sum(fehler_neu):.0f}s")
    print("\nLesart: Zeile 2 hoch + Zeile 1 niedrig = die Fehler sind "
          "FALSCHE Anker (Semantik, z. B. Trailer-Zonen), kein fehlendes "
          "Signal — dann braucht es Anker-AUSWAHL, nicht Anker-Suche. "
          "Zeile 3 ist die Untergrenze dessen, was ein dummer Snap holt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
