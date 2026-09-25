#!/usr/bin/env python3
"""O26 — Hebt OCR (Bildschirm-Text) als Zusatzspalte die binaere Leistung?

DIE FRAGE
---------
Der Backbone sieht 224x224 und kann eingeblendeten Text nicht lesen (Memory
backbone_liest_keinen_text). Die flaechendeckende OCR-Spur (tv-ocr-spur,
2026-09-24, 322 Aufnahmen) holt diese Information zurueck. Gemessen am
Messsatz, ohne Training: NN-Fehlersekunden liegen 5,5-mal so oft wie
richtige in +-10 s eines OCR-Treffers, und die Anreicherung haelt nach
Abstand zur Label-Kante getrennt (1,6x bis 5,1x). Das ist ein Zusammenhang.
Ob ein Kopf ihn NUTZEN kann, misst dieser Lauf.

DIE SPALTEN (drei, hinten angehaengt, je Sekunde)
-------------------------------------------------
  hinweis_nah  1, wenn ein Programmhinweis (Wochentag + Uhrzeit) in +-10 s liegt
  werbung_nah  1, wenn eine Werbe-Kennzeichnung in +-10 s liegt
  spur_da      1, wenn die Aufnahme eine Spur hat und die Sekunde abgetastet ist
Ohne spur_da waere "kein Text" von "nie hingesehen" nicht zu trennen: 71 %
der train-Aufnahmen haben keine Spur (Quelle weg).

WAS DIE ARME TRENNT
-------------------
Nur die drei Spalten. Dieselben Zeilen (train mit Schrittweite 4, test
vollstaendig), dieselben Seeds, dieselbe Architektur, dieselbe
Standardisierung (Kennwerte nur aus train).

METRIK
------
Primaer: F1 (geglaettet 10 s je Aufnahme) auf den test-Aufnahmen MIT Spur —
nur dort kann die Spalte wirken. Neben: F1 auf allen test-Aufnahmen.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HIER = Path(__file__).resolve().parent
SPUR = Path.home() / ".cache/tvd-ocr-spur"
FENSTER = 10  # +- Sekunden, aus der Anreicherungsanalyse vom 24.09.


def _o20():
    spec = importlib.util.spec_from_file_location("o20", _HIER / "o20-klassen-split.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ocr_spalten(uuid, n_sek):
    """(n_sek, 3) je Sekunde: hinweis_nah, werbung_nah, spur_da. Die
    Rechnung steht seit 2026-09-25 in ocr_spalten.py (eine Definition fuer
    Training, Messung und die Go-Paritaet)."""
    return _oc().ocr_spalten(uuid, n_sek, spur_dir=SPUR)


def _oc():
    spec = importlib.util.spec_from_file_location("ocr_spalten", _HIER / "ocr_spalten.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def spalten_fuer(rec, uuids, schritt):
    """OCR-Spalten zeilengenau passend zu X: je Aufnahme die Sekunden
    0, schritt, 2*schritt, ... (so unterabtastet lade())."""
    aus = np.zeros((len(rec), 3), np.float32)
    for i, u in enumerate(uuids):
        m = rec == i
        k = int(m.sum())
        voll = ocr_spalten(u, k * schritt + schritt)
        aus[m] = voll[::schritt][:k]
    return aus


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--nur-kontrollarm", action="store_true", dest="nur_kontroll")
    ap.add_argument("--json")
    a = ap.parse_args()

    o = _o20()
    print("Lade train …", flush=True)
    Xtr, ytr, rec_tr, u_tr = o.lade("train", None, a.schritt, mit_uuids=True)
    print("Lade test …", flush=True)
    Xte, yte, rec_te, u_te = o.lade("test", None, 1, mit_uuids=True)

    Otr = spalten_fuer(rec_tr, u_tr, a.schritt)
    Ote = spalten_fuer(rec_te, u_te, 1)
    mit_spur = np.array([(SPUR / f"{u}.json").is_file() for u in u_te])
    maske_spur = mit_spur[rec_te]
    print(f"train {len(u_tr)} Aufnahmen ({int(sum((SPUR / f'{u}.json').is_file() for u in u_tr))} "
          f"mit Spur), test {len(u_te)} ({int(mit_spur.sum())} mit Spur)", flush=True)
    print(f"test-Zeilen mit hinweis_nah {Ote[:, 0].mean():.3f}, werbung_nah "
          f"{Ote[:, 1].mean():.3f}, spur_da {Ote[:, 2].mean():.3f}", flush=True)

    Xtr_s, Xte_s = o.standardisieren(Xtr, Xte)
    Xtr_ps, Xte_ps = o.standardisieren(np.concatenate([Xtr, Otr], 1),
                                       np.concatenate([Xte, Ote], 1))
    wahr = (yte > 0).astype(np.int64)

    def werte(p):
        pred = (o.glaetten(p, rec_te) > 0.5).astype(np.int64)
        return (o.f1(pred[maske_spur], wahr[maske_spur]), o.f1(pred, wahr))

    erg = {"ohne": [], "mit": []}
    arme = ["ohne"] if a.nur_kontroll else ["ohne", "mit"]
    for seed in range(a.seeds):
        for arm in arme:
            A, B = (Xtr_s, Xte_s) if arm == "ohne" else (Xtr_ps, Xte_ps)
            p = o.fit_und_werte(A, ytr, B, yte, rec_te, 2, seed, a.epochen, a.hidden)
            primaer, alle = werte(p)
            erg[arm].append([primaer, alle])
            print(f"  Seed {seed}  {arm:<5} F1 mit Spur {primaer:.4f}  alle {alle:.4f}", flush=True)
    for arm in arme:
        v = np.array(erg[arm])[:, 0]
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        print(f"\n{arm}: primaer Median {np.median(v):.4f}  sd {sd:.4f}")
    if not a.nur_kontroll:
        d = np.array(erg["mit"])[:, 0] - np.array(erg["ohne"])[:, 0]
        d2 = np.array(erg["mit"])[:, 1] - np.array(erg["ohne"])[:, 1]
        print(f"\nDelta primaer (mit - ohne), gepaart: Median {np.median(d):+.4f}  "
              f"positiv in {int((d > 0).sum())} von {len(d)} Seeds")
        print(f"Delta alle test (Nebenwert): Median {np.median(d2):+.4f}")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
