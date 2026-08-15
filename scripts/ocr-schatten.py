#!/usr/bin/env python3
"""O13-Schattenlauf: was HÄTTE die OCR-Regel an den Kanten getan?

Registrierung: docs/o13-ocr-schatten-preregistration.md.

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
LEDGER = ARCHIV / "ocr-schatten.jsonl"
DUMPS = Path.home() / ".cache/tv-detect-daemon/emit-signals"
SNAPSHOT = Path("/tmp/tv-train-snapshot")

# Der Schnitt. Labels, die davor gesetzt wurden, zählen NICHT — sie sind
# potentiell dieselben, an denen O9–O12 gemessen haben.
AKTIVIERUNG = 1786804904  # 2026-08-15 16:41

# Die Regel, fixiert in der Registrierung.
FENSTER_S = 90.0
NACHLAUF_S = 5.0

# Registrierte Auswertungs-Schwellen. Vorher wird nicht hineingeschaut.
MIN_KANTEN = 40
MIN_AUFNAHMEN = 20
SENKUNG_MIND = 0.25     # Summe Kantenfehler auf angefassten Kanten
VERHAELTNIS_MIND = 2.0  # besser : schlechter
EINZELVERLUST_MAX = 30.0


def bloecke(pfad):
    try:
        d = json.loads(Path(pfad).read_text())
    except Exception:
        return None
    b = d.get("ads")
    if not b:
        return None
    return sorted((float(x[0]), float(x[1])) for x in b)


def reviewed_at(pfad):
    try:
        return float(json.loads(Path(pfad).read_text()).get("reviewed_at") or 0)
    except Exception:
        return 0.0


def ist_mensch(pfad):
    """Wie hasProtectedLabels in tv-recorder: auto_confirmed_at trennt
    Maschine von Mensch, die blosse Existenz der Datei nicht."""
    try:
        d = json.loads(Path(pfad).read_text())
    except Exception:
        return False
    return not d.get("auto_confirmed_at")


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
            if not up.is_file() or not ist_mensch(up):
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
            if "ocr_funde" not in j:
                continue  # Dump ohne OCR — nicht auswertbar, nicht mitzählen
            user = bloecke(up)
            auto = bloecke(d / "ads.json")
            if not user or not auto:
                continue
            vorschlag, angefasst = regel_kanten(auto, j.get("ocr_funde") or [])
            zeilen = []
            for ub in user:
                paare = [(ueberlappung(ub, ab), i) for i, ab in enumerate(auto)]
                paare = [p for p in paare if p[0] > 0]
                if not paare:
                    continue
                _, i = max(paare)
                for seite, wahr, ist, soll in (
                        ("start", ub[0], auto[i][0], vorschlag[i][0]),
                        ("ende", ub[1], auto[i][1], vorschlag[i][1])):
                    zeilen.append({
                        "seite": seite,
                        "fehler_ist": round(abs(ist - wahr), 2),
                        "fehler_regel": round(abs(soll - wahr), 2),
                        "angefasst": (i, seite) in angefasst,
                    })
            if not zeilen:
                continue
            f.write(json.dumps({
                "uuid": u, "reviewed_at": reviewed_at(up),
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
    ang = [(e["uuid"], k) for e in eintraege for k in e["kanten"] if k["angefasst"]]
    n_auf = len({u for u, _ in ang})
    print(f"{len(eintraege)} Aufnahmen im Ledger, {len(ang)} angefasste Kanten "
          f"aus {n_auf} Aufnahmen")
    # ⚠️ Vor Erreichen der registrierten Mindestmengen wird NICHT ausgewertet.
    # Ein Zwischenstand, den man anschauen darf, ist keine Vorab-Registrierung
    # mehr — man hört auf, sobald die Zahl gefällt.
    if len(ang) < MIN_KANTEN or n_auf < MIN_AUFNAHMEN:
        print(f"  Noch nicht auswertbar: verlangt sind {MIN_KANTEN} Kanten "
              f"aus {MIN_AUFNAHMEN} Aufnahmen. Kein Zwischenstand.")
        return 0
    v = sum(k["fehler_ist"] for _, k in ang)
    n = sum(k["fehler_regel"] for _, k in ang)
    bes = sum(1 for _, k in ang if k["fehler_regel"] < k["fehler_ist"] - 0.5)
    sch = sum(1 for _, k in ang if k["fehler_regel"] > k["fehler_ist"] + 0.5)
    verl = max((k["fehler_regel"] - k["fehler_ist"] for _, k in ang), default=0.0)
    senkung = (v - n) / v if v else 0.0
    verh = bes / sch if sch else float("inf")
    b1, b2, b3 = (senkung >= SENKUNG_MIND, verh >= VERHAELTNIS_MIND,
                  verl <= EINZELVERLUST_MAX)
    print(f"\n  [1] Kantenfehler {v:.0f}s -> {n:.0f}s  ({senkung:+.1%})   "
          f"{'ERFUELLT' if b1 else 'VERFEHLT'} (>= {SENKUNG_MIND:.0%})")
    print(f"  [2] besser {bes} : schlechter {sch} = {verh:.1f}   "
          f"{'ERFUELLT' if b2 else 'VERFEHLT'} (>= {VERHAELTNIS_MIND:.1f})")
    print(f"  [3] groesster Einzelverlust {verl:.0f}s   "
          f"{'ERFUELLT' if b3 else 'VERFEHLT'} (<= {EINZELVERLUST_MAX:.0f}s)")
    print(f"\n  ==> O13 {'ERFUELLT' if (b1 and b2 and b3) else 'NICHT ERFUELLT'}")
    alle = [k for e in eintraege for k in e["kanten"]]
    print(f"\n  Nachrichtlich: {len(alle)} Kanten gesamt, "
          f"davon {100*len(ang)/len(alle):.0f} % angefasst; "
          f"Anteil <= 2 s {100*sum(1 for k in alle if k['fehler_ist']<=2)/len(alle):.0f} % -> "
          f"{100*sum(1 for k in alle if (k['fehler_regel'] if k['angefasst'] else k['fehler_ist'])<=2)/len(alle):.0f} %")
    return 0


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
