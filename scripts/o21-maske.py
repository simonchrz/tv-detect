#!/usr/bin/env python3
"""O21 — vergiften widersprüchliche Frames das Training?

DIE FRAGE
---------
Das Fehlerbudget (Ledger, dreizehnter Durchgang) ordnet **39 % des
IoU-Verlusts** der Label-Seite zu: Sekunden, an denen das Label harter
Evidenz widerspricht. Das ist kein Reviewfehler — Simon reviewt
vollständig. Es ist eine Konvention: der Review beantwortet „was muss der
Spieler überspringen", das Training liest die Antwort als „was ist
Werbung". An Trailern, Idents, Split-Screen und den Rändern fallen beide
auseinander.

Statt die Labels umzuschreiben (L2, und die Konvention ist für ihren
Zweck richtig) werden die widersprüchlichen Frames aus dem TRAINING
genommen. Der Code kennt das Muster: `frame_mask` in train-head.py, mit
dem Kommentar „no opinion = not training data, NOT a default-show
prediction".

WAS EIN WIDERSPRUCH IST — nur HARTE Evidenz
--------------------------------------------
  * Ein Wiederholungs-Anker deckt die Sekunde, das Label sagt Sendung.
    Die Anker haben 96.8 % Präzision gegen Menschenlabel (geeicht
    2026-09-07). Ein wiederholter Zusammenschnitt IST Werbung.
  * Die Sekunde liegt im Endblock (Block endet am Aufnahmeende, >= 60 s),
    den das Label als Werbung führt. Von 54 solchen Blöcken sind 12 die
    Folgesendung; die Form allein trennt sie nicht, deshalb zählt hier
    nur, dass das Label dort unzuverlässig IST.

Das Senderlogo wäre der dritte Zeuge (858 s „Label vermutlich zu breit"
im Budget), ist aber schwächer: Let's Dance blendet es aus, sixx wäscht
es aus. Es bleibt bewusst draußen.

WIE GEMESSEN WIRD — und was das nicht zeigt
--------------------------------------------
Beide Arme werden auf demselben UNBESTRITTENEN Teil des Testsatzes
gemessen, also dort, wo Label und Evidenz sich nicht widersprechen. Gegen
die vollen Labels zu messen hiesse, gegen genau die Labels zu messen, die
unter Verdacht stehen — der maskierte Arm würde dafür bestraft, dass er
Trailer nicht mehr als Werbung lernt.

⚠️ Der unbestrittene Teil schliesst die strittigen Fälle per
Konstruktion aus. Ein positives Ergebnis heisst deshalb „das Modell wird
auf dem unstrittigen Teil besser", NICHT „das Modell wird insgesamt
besser". Ein Null-Ergebnis dagegen erledigt die Vergiftungs-These.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

_HIER = Path(__file__).resolve().parent
ARCH = Path.home() / ".cache" / "tvd-train-archive"
BILD = Path.home() / ".cache" / "tvd-wiederholung" / "korpus"


def _o20():
    spec = importlib.util.spec_from_file_location("o20", _HIER / "o20-klassen-split.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def widerspruch(eimer, limit=None, schritt=1):
    """Maske je Zeile: True = Label widerspricht harter Evidenz.

    Muss dieselbe Reihenfolge und dieselbe Unterabtastung erzeugen wie
    o20.lade(), sonst zeigt die Maske auf fremde Zeilen — genau die
    Klasse Fehler, die heute schon einmal vier Funde erfunden hat
    (frames_tragen_erwartete_zeit).
    """
    led = json.loads((ARCH / "split-ledger.json").read_text())
    ziel = sorted(u for u, v in led.items() if v == eimer)
    if limit:
        ziel = ziel[:limit]
    aus = []
    for u in ziel:
        f = ARCH / f"{u}.npz"
        if not f.is_file():
            continue
        try:
            m = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        except Exception:
            continue
        fp = m.get("feature_npy", "")
        ads = m.get("ads") or []
        if not fp or not os.path.exists(fp) or not ads:
            continue
        F = np.load(fp, mmap_mode="r")
        if F.shape[1] not in (1281, 1282):
            continue
        n = F.shape[0]
        y = np.zeros(n, bool)
        for a, b in ads:
            y[max(0, int(a)):min(n, int(b))] = True
        w = np.zeros(n, bool)
        # (1) Anker deckt, Label sagt Sendung.
        p = BILD / f"{u}.json"
        if p.is_file():
            try:
                A = np.zeros(n, bool)
                for x in json.loads(p.read_text())["anchored"]:
                    A[max(0, int(x["window_start_s"])):min(n, int(x["end_s"]) + 1)] = True
                w |= (A & ~y)
            except Exception:
                pass
        # (2) Endblock am Aufnahmeende.
        for a, b in ads:
            if n - float(b) <= 3 and float(b) - float(a) >= 60:
                w[max(0, int(a)):min(n, int(b))] = True
        aus.append(w[::schritt] if schritt > 1 else w)
    return np.concatenate(aus) if aus else np.zeros(0, bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochen", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--schritt", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--nur-kontrollarm", action="store_true", dest="nur_kontroll")
    ap.add_argument("--json")
    a = ap.parse_args()

    o = _o20()
    print("Lade train …", flush=True)
    Xtr, ytr, _ = o.lade("train", a.limit, a.schritt)
    wtr = widerspruch("train", a.limit, a.schritt)
    print("Lade test …", flush=True)
    Xte, yte, rec_te = o.lade("test", a.limit, 1)
    wte = widerspruch("test", a.limit, 1)
    if Xtr is None or Xte is None:
        print("keine Daten"); return 1
    assert len(wtr) == Xtr.shape[0], f"Maske train {len(wtr)} != Zeilen {Xtr.shape[0]}"
    assert len(wte) == Xte.shape[0], f"Maske test {len(wte)} != Zeilen {Xte.shape[0]}"
    print(f"train {Xtr.shape[0]} Zeilen, davon widersprüchlich {int(wtr.sum())} "
          f"({100*wtr.mean():.1f} %)")
    print(f"test  {Xte.shape[0]} Zeilen, davon widersprüchlich {int(wte.sum())} "
          f"({100*wte.mean():.1f} %)")
    Xtr, Xte = o.standardisieren(Xtr, Xte)
    wahr = (yte > 0).astype(np.int64)
    # Gemessen wird NUR auf dem unbestrittenen Teil, fuer BEIDE Arme gleich.
    mess = ~wte
    print(f"gemessen auf {int(mess.sum())} unbestrittenen Testzeilen")

    erg = {"alle": [], "maskiert": []}
    arme = ["alle"] if a.nur_kontroll else ["alle", "maskiert"]
    for seed in range(a.seeds):
        for arm in arme:
            m = np.ones(len(ytr), bool) if arm == "alle" else ~wtr
            p = o.fit_und_werte(Xtr[m], ytr[m], Xte, yte, rec_te, 2, seed,
                                a.epochen, a.hidden)
            ps = o.glaetten(p, rec_te)
            s = o.f1((ps[mess] > 0.5).astype(np.int64), wahr[mess])
            erg[arm].append(s)
            print(f"  Seed {seed}  {arm:<9} F1 {s:.4f}", flush=True)
    for arm in arme:
        v = np.array(erg[arm])
        print(f"\n{arm}: Median {np.median(v):.4f}  Mittel {v.mean():.4f}  "
              f"sd {v.std(ddof=1) if len(v)>1 else float('nan'):.4f}")
    if not a.nur_kontroll:
        d = np.array(erg["maskiert"]) - np.array(erg["alle"])
        print(f"\nDelta (maskiert minus alle), gepaart: Median {np.median(d):+.4f}  "
              f"positiv in {int((d>0).sum())} von {len(d)} Seeds")
    if a.json:
        Path(a.json).write_text(json.dumps(erg, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
