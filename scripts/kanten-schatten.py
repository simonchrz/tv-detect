#!/usr/bin/env python3
"""Kanten-Schattenlauf: was HÄTTEN die Kandidaten-Regeln getan?

Zwei Regeln, gemessen auf denselben Kanten, jede mit eigener Registrierung:

  * **OCR** (O13, docs/o13-ocr-schatten-preregistration.md) — Blockgrenze
    an einen Programmhinweis oder eine Werbe-Kennzeichnung ziehen.
  * **Flanke** (O14, docs/o14-flankenauswahl-preregistration.md) — von
    NN-Flanke und Logo-Flanke die mit der groesseren Amplitude waehlen.
    AUSWAHL statt Mischung: hsmm-blend mittelt beide und verliert dabei
    beide, weil sie in verschiedenen Faellen recht haben.

⚠️ Prospektiv, und das ist der ganze Punkt. Vier Vorgänger (O9–O12) haben
dieselbe Idee auf handverlesenen Sätzen geprüft und sind an vier
verschiedenen Bedingungen gescheitert; der Golden-Satz ist seit O12 für
diese Regel verbraucht. Ein fünfter Satz wäre die Manipulation, gegen die
Vorab-Registrierungen existieren. Also wird ab AKTIVIERUNG nur noch auf
Aufnahmen gesammelt, deren Labels danach entstanden sind — die kann
niemand rückwirkend passend auswählen.

Die Regel wird NICHT angewandt. Dieses Skript schreibt nur mit.

Aufruf (nächtlich):  python3 scripts/ocr-schatten.py
Auswertung:          python3 scripts/ocr-schatten.py --auswerten
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

ARCHIV = Path.home() / ".cache/tvd-train-archive"
LEDGER = ARCHIV / "kanten-schatten.jsonl"
DUMPS = Path.home() / ".cache/tv-detect-daemon/emit-signals"
SNAPSHOT = Path("/tmp/tv-train-snapshot")

# Der Schnitt. Labels, die davor gesetzt wurden, zählen NICHT — sie sind
# potentiell dieselben, an denen O9–O12 gemessen haben.
AKTIVIERUNG = 1786804904  # 2026-08-15 16:41

# O13 (OCR), fixiert in der Registrierung.
FENSTER_S = 90.0
NACHLAUF_S = 5.0

# O14 (Flankenauswahl), fixiert in der Registrierung.
FLANKE_FENSTER_S = 30.0
FLANKE_GLAETTUNG_S = 10
FLANKE_AMPLITUDE_S = 6

# Registrierte Auswertungs-Schwellen. Vorher wird nicht hineingeschaut.
MIN_KANTEN = 40
MIN_AUFNAHMEN = 20
SENKUNG_MIND = 0.25     # Summe Kantenfehler auf angefassten Kanten
VERHAELTNIS_MIND = 2.0  # besser : schlechter
EINZELVERLUST_MAX = 30.0
O14_MIN_KANTEN = 60   # O14 fasst fast jede Kante an, nicht nur die mit Marker


def bloecke(pfad):
    """Blöcke aus ads.json ODER ads_user.json.

    ⚠️ Die beiden Dateien haben NICHT dasselbe Format: `ads_user.json` ist
    ein Objekt mit "ads", `ads.json` eine nackte Liste von Paaren. Die
    erste Fassung las nur das Objekt — der Schattenlauf hätte damit jede
    Aufnahme still übersprungen und für immer nichts gesammelt. Genau so
    sieht ein Mechanismus aus, der monatelang „läuft" und nie ein Ergebnis
    hat.
    """
    try:
        d = json.loads(Path(pfad).read_text())
    except Exception:
        return None
    b = d.get("ads") if isinstance(d, dict) else d
    if not b:
        return None
    try:
        return sorted((float(x[0]), float(x[1])) for x in b)
    except (TypeError, IndexError, ValueError):
        return None


def reviewed_at(pfad):
    try:
        return float(json.loads(Path(pfad).read_text()).get("reviewed_at") or 0)
    except Exception:
        return 0.0


def label_quelle(pfad):
    """"mensch" | "agent" | "auto" — WER das Label gesetzt hat.

    ⚠️ Die Unterscheidung ist nicht kosmetisch, sie entscheidet ueber die
    Gueltigkeit von O13. Die OCR-Regel setzt die Kante an einen
    Programmhinweis; ein Agent, der denselben Frame sieht, liest „Montag
    20:15" und setzt die Kante genau dort. Gegen Agent-Labels geprueft
    wuerde die Regel ihre eigene Evidenz erzeugen und trivial bestehen.

    O14 (Flankenauswahl aus NN und Logo) ist davon NICHT betroffen — diese
    Signale sieht kein Agent, sein Urteil ist dazu unabhaengig.

    Erkennung: auto_confirmed_at = Maschine. reviewed_by gesetzt = von
    einem Review-Agenten. Alles andere = Mensch (der Zustand aller
    Altbestaende — die App schickt kein reviewed_by).
    """
    try:
        d = json.loads(Path(pfad).read_text())
    except Exception:
        return None
    if d.get("auto_confirmed_at"):
        return "auto"
    if d.get("reviewed_by"):
        return "agent"
    return "mensch"


def regel_kanten(auto, funde):
    """Die Regel auf die Auto-Blöcke anwenden — nur nach außen, nie kürzen."""
    if not funde:
        return list(auto), set()
    hin = [f for f in funde if f.get("hinweis") or f.get("werbemarker")]
    neu, angefasst = [], set()
    for i, (s, e) in enumerate(auto):
        ns, ne = s, e
        # Ende: Treffer im Fenster dahinter, Flüchtigkeit über das
        # Referenzfenster (der Dump enthält nur zählende Funde, die
        # Dauer-Einblendungen sind beim Erheben schon rausgefallen).
        nach = [f["time_s"] for f in hin if e < f["time_s"] <= e + FENSTER_S]
        if nach:
            ne = max(nach) + NACHLAUF_S
            angefasst.add((i, "ende"))
        vor = [f["time_s"] for f in hin if s - FENSTER_S <= f["time_s"] < s]
        if vor:
            ns = min(vor) - NACHLAUF_S
            angefasst.add((i, "start"))
        neu.append((ns, ne))
    return neu, angefasst


def ueberlappung(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _persekunde(x, fps):
    n = len(x) // fps
    return [sum(x[i*fps:(i+1)*fps]) / fps for i in range(n)]


def _glatt(v, w):
    out, n = [], len(v)
    for i in range(n):
        a, b = max(0, i - w // 2), min(n, i + w // 2 + 1)
        out.append(sum(v[a:b]) / (b - a))
    return out


def _flanke(sig, t, fenster):
    """Naechster Schwellwert-Durchgang (0.5) um t, plus lokale Amplitude."""
    a, b = max(0, int(t - fenster)), min(len(sig) - 1, int(t + fenster))
    if b <= a:
        return None, 0.0
    ueber = [sig[i] > 0.5 for i in range(a, b)]
    wechsel = [a + i + 0.5 for i in range(len(ueber) - 1)
               if ueber[i] != ueber[i + 1]]
    if not wechsel:
        return None, 0.0
    x = min(wechsel, key=lambda z: abs(z - t))
    lo, hi = max(0, int(x - FLANKE_AMPLITUDE_S)), min(len(sig), int(x + FLANKE_AMPLITUDE_S))
    fenster_werte = sig[lo:hi]
    return x, (max(fenster_werte) - min(fenster_werte)) if fenster_werte else 0.0


def flanken_kanten(auto, dump):
    """O14: je Kante die Flanke mit der groesseren Amplitude waehlen.

    Gibt (neue Kanten, welches Signal gewaehlt wurde) zurueck. Ohne Flanke
    im Fenster bleibt die Kante unveraendert — die Regel setzt nie eine
    Kante, fuer die es keinen Beleg gibt.
    """
    nn = dump.get("nn_confs") or []
    lg = dump.get("logo_confs") or []
    fps = int(dump.get("fps") or 0)
    if not nn or len(lg) != len(nn) or fps <= 0:
        return list(auto), {}
    ns = _glatt(_persekunde(nn, fps), FLANKE_GLAETTUNG_S)
    ls = _glatt(_persekunde([1.0 - x for x in lg], fps), FLANKE_GLAETTUNG_S)
    neu, wahl = [], {}
    for i, (s0, e0) in enumerate(auto):
        kante = []
        for seite, t in (("start", s0), ("ende", e0)):
            fn, an = _flanke(ns, t, FLANKE_FENSTER_S)
            fl, al = _flanke(ls, t, FLANKE_FENSTER_S)
            if fn is None and fl is None:
                kante.append(t); wahl[(i, seite)] = "keine"; continue
            if fl is not None and (fn is None or al > an):
                kante.append(fl); wahl[(i, seite)] = "logo"
            else:
                kante.append(fn); wahl[(i, seite)] = "nn"
        neu.append((kante[0], kante[1]))
    return neu, wahl


def sammle():
    gesehen = set()
    if LEDGER.exists():
        for z in LEDGER.read_text().splitlines():
            if z.strip():
                try:
                    gesehen.add(json.loads(z)["uuid"])
                except Exception:
                    pass
    neu = 0
    with open(LEDGER, "a") as f:
        for d in sorted(SNAPSHOT.glob("_rec_*")):
            u = d.name[5:]
            if u in gesehen:
                continue
            up = d / "ads_user.json"
            quelle = label_quelle(up) if up.is_file() else None
            if quelle in (None, "auto"):
                continue
            # ⚠️ DER Filter: nur Labels, die nach der Aktivierung entstanden.
            if reviewed_at(up) < AKTIVIERUNG:
                continue
            dump = DUMPS / f"{u}.json"
            if not dump.is_file():
                continue
            try:
                j = json.loads(dump.read_text())
            except Exception:
                continue
            # ⚠️ Ein Dump OHNE ocr_funde ist fuer O13 unbrauchbar, fuer O14
            # aber vollstaendig: die Flankenauswahl braucht nur nn_confs und
            # logo_confs. Die erste Fassung sprang hier ganz raus und haette
            # O14 an einer Voraussetzung verhungern lassen, die es gar nicht
            # hat. Stattdessen wird die Aufnahme aufgenommen und O13 fuer sie
            # als "nicht angefasst" gefuehrt.
            hat_ocr = "ocr_funde" in j
            user = bloecke(up)
            auto = bloecke(d / "ads.json")
            if not user or not auto:
                continue
            vorschlag, angefasst = (regel_kanten(auto, j.get("ocr_funde") or [])
                                    if hat_ocr else (list(auto), set()))
            fl_vorschlag, fl_wahl = flanken_kanten(auto, j)
            zeilen = []
            for ub in user:
                paare = [(ueberlappung(ub, ab), i) for i, ab in enumerate(auto)]
                paare = [p for p in paare if p[0] > 0]
                if not paare:
                    continue
                _, i = max(paare)
                for seite, wahr, ist, soll, fl in (
                        ("start", ub[0], auto[i][0], vorschlag[i][0], fl_vorschlag[i][0]),
                        ("ende", ub[1], auto[i][1], vorschlag[i][1], fl_vorschlag[i][1])):
                    zeilen.append({
                        "seite": seite,
                        "fehler_ist": round(abs(ist - wahr), 2),
                        "fehler_ocr": round(abs(soll - wahr), 2),
                        "ocr_angefasst": (i, seite) in angefasst,
                        "fehler_flanke": round(abs(fl - wahr), 2),
                        "flanke_wahl": fl_wahl.get((i, seite), "keine"),
                    })
            if not zeilen:
                continue
            f.write(json.dumps({
                "uuid": u, "reviewed_at": reviewed_at(up),
                "label_quelle": quelle,
                "n_ocr_funde": len(j.get("ocr_funde") or []),
                "kanten": zeilen}) + "\n")
            neu += 1
    print(f"Schattenlauf: {neu} Aufnahmen ergänzt "
          f"(Labels ab {AKTIVIERUNG}, Dumps mit OCR).")


def auswerten():
    if not LEDGER.exists():
        print("Noch keine Schatten-Zeilen.")
        return 0
    eintraege = [json.loads(z) for z in LEDGER.read_text().splitlines() if z.strip()]
    mit_q = [(e["uuid"], k, e.get("label_quelle", "mensch"))
             for e in eintraege for k in e["kanten"]]
    alle = [(u, k) for u, k, _ in mit_q]
    import collections
    print(f"{len(eintraege)} Aufnahmen im Ledger, {len(alle)} Kanten "
          f"(Label-Herkunft: {dict(collections.Counter(q for _,_,q in mit_q))})\n")
    _auswerten_o13(mit_q)
    print()
    _auswerten_o14(alle)
    return 0


def _auswerten_o13(alle_mit_quelle):
    # ⚠️ NUR Menschen-Labels. Gegen Agent-Labels waere die Frage zirkulaer
    # (s. label_quelle): der Agent liest denselben Bildschirm-Text, an den
    # die Regel die Kante zieht.
    alle = [(u, k) for u, k, q in alle_mit_quelle if q == "mensch"]
    ang = [(u, k) for u, k in alle if k["ocr_angefasst"]]
    n_auf = len({u for u, _ in ang})
    print(f"O13 (OCR): {len(ang)} angefasste Kanten aus {n_auf} Aufnahmen")
    # ⚠️ Vor den registrierten Mindestmengen wird NICHT ausgewertet. Ein
    # Zwischenstand, den man anschauen darf, ist keine Vorab-Registrierung
    # mehr — man hoert auf, sobald die Zahl gefaellt.
    if len(ang) < MIN_KANTEN or n_auf < MIN_AUFNAHMEN:
        print(f"  Noch nicht auswertbar (verlangt {MIN_KANTEN} Kanten aus "
              f"{MIN_AUFNAHMEN} Aufnahmen). Kein Zwischenstand.")
        return
    v = sum(k["fehler_ist"] for _, k in ang)
    n = sum(k["fehler_ocr"] for _, k in ang)
    bes = sum(1 for _, k in ang if k["fehler_ocr"] < k["fehler_ist"] - 0.5)
    sch = sum(1 for _, k in ang if k["fehler_ocr"] > k["fehler_ist"] + 0.5)
    verl = max((k["fehler_ocr"] - k["fehler_ist"] for _, k in ang), default=0.0)
    senkung = (v - n) / v if v else 0.0
    verh = bes / sch if sch else float("inf")
    b = (senkung >= SENKUNG_MIND, verh >= VERHAELTNIS_MIND, verl <= EINZELVERLUST_MAX)
    print(f"  [1] Kantenfehler {v:.0f}s -> {n:.0f}s ({senkung:+.1%})  "
          f"{'ERFUELLT' if b[0] else 'VERFEHLT'}")
    print(f"  [2] besser {bes} : schlechter {sch} = {verh:.1f}  "
          f"{'ERFUELLT' if b[1] else 'VERFEHLT'}")
    print(f"  [3] groesster Einzelverlust {verl:.0f}s  "
          f"{'ERFUELLT' if b[2] else 'VERFEHLT'}")
    print(f"  ==> O13 {'ERFUELLT' if all(b) else 'NICHT ERFUELLT'}")


def _auswerten_o14(alle):
    n_auf = len({u for u, _ in alle})
    print(f"O14 (Flankenauswahl): {len(alle)} Kanten aus {n_auf} Aufnahmen")
    if len(alle) < O14_MIN_KANTEN or n_auf < MIN_AUFNAHMEN:
        print(f"  Noch nicht auswertbar (verlangt {O14_MIN_KANTEN} Kanten aus "
              f"{MIN_AUFNAHMEN} Aufnahmen). Kein Zwischenstand.")
        return
    ist = sorted(k["fehler_ist"] for _, k in alle)
    neu = sorted(k["fehler_flanke"] for _, k in alle)
    med = lambda a: a[len(a)//2]
    d_med = med(ist) - med(neu)
    p_ist = 100 * sum(1 for x in ist if x <= 2) / len(ist)
    p_neu = 100 * sum(1 for x in neu if x <= 2) / len(neu)
    s_ist, s_neu = sum(ist), sum(neu)
    b = (d_med >= 0.5, p_neu - p_ist >= 3.0, s_neu <= s_ist)
    print(f"  [1] Median {med(ist):.1f}s -> {med(neu):.1f}s ({d_med:+.1f}s)  "
          f"{'ERFUELLT' if b[0] else 'VERFEHLT'} (>= 0.5s)")
    print(f"  [2] <=2s {p_ist:.0f}% -> {p_neu:.0f}% ({p_neu-p_ist:+.0f} Punkte)  "
          f"{'ERFUELLT' if b[1] else 'VERFEHLT'} (>= 3)")
    print(f"  [3] Gesamtfehler {s_ist:.0f}s -> {s_neu:.0f}s "
          f"({100*(s_neu-s_ist)/max(s_ist,1):+.0f} %)  "
          f"{'ERFUELLT' if b[2] else 'VERFEHLT'} (<= 0 %)")
    import collections
    w = collections.Counter(k["flanke_wahl"] for _, k in alle)
    print(f"  gewaehlt: {dict(w)}")
    print(f"  ==> O14 {'ERFUELLT' if all(b) else 'NICHT ERFUELLT'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--auswerten", action="store_true")
    a = ap.parse_args()
    ARCHIV.mkdir(parents=True, exist_ok=True)
    if a.auswerten:
        return auswerten()
    sammle()
    return 0


if __name__ == "__main__":
    sys.exit(main())
