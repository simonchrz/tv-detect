#!/usr/bin/env python3
"""Prüft eine Vorab-Registrierung gegen die tatsächliche Serie.

Der Punkt dieses Skripts ist NICHT, Arbeit zu sparen. Es ist die einzige
Instanz in der Schleife, die zu einem Ergebnis "NICHT ERFUELLT" sagen kann,
ohne dass derjenige mitredet, der die Registrierung geschrieben hat. Wer eine
Hypothese aufstellt, die Daten sammelt UND am Ende beurteilt, ob sie hielt,
hat keine Prüfung, sondern eine Erzählung.

Es liest die Regel aus dem ```regel-Block der Registrierung, filtert die
gültigen Nächte aus shadow-trend.jsonl und rechnet nach. Zusätzlich prüft es
über die git-Historie, ob die Regel NACH der ersten gezählten Nacht noch
angefasst wurde — das ist der Weg, auf dem eine Vorab-Registrierung still zu
einer Nachher-Registrierung wird.

    audit-preregistration.py [--archiv PFAD] [--docs PFAD]

Rückgabewert: 0 = alle Serien noch offen oder erfüllt, 1 = mindestens eine
Regel verletzt oder Integritätsproblem. Damit taugt es als cron-Wächter.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ARCHIV = Path.home() / ".cache/tvd-train-archive"
DOCS = Path(__file__).resolve().parent.parent / "docs"


def regeln_laden(docs):
    """Alle ```regel-Blöcke aus den Registrierungen."""
    out = []
    for pfad in sorted(docs.glob("*preregistration*.md")):
        for m in re.finditer(r"^```regel\n(.*?)^```", pfad.read_text(),
                             flags=re.M | re.S):
            try:
                out.append((pfad, json.loads(m.group(1))))
            except json.JSONDecodeError as e:
                print(f"  ⚠ {pfad.name}: regel-Block nicht lesbar ({e})")
    return out


def naechte_laden(archiv):
    """ts → {arch: zeile} aus shadow-trend.jsonl."""
    pfad = archiv / "shadow-trend.jsonl"
    if not pfad.exists():
        return {}
    nach_ts = defaultdict(dict)
    for ln in pfad.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            e = json.loads(ln)
        except json.JSONDecodeError:
            continue
        nach_ts[e.get("ts")][e.get("arch")] = e
    return nach_ts


def regel_zuletzt_geaendert(pfad):
    """Unix-Zeit des letzten Commits, der die Datei angefasst hat.

    Ungetrackte oder uncommittete Änderungen zählen als 'jetzt' — eine Regel,
    die nur im Arbeitsverzeichnis steht, ist nicht festgeschrieben.
    """
    try:
        r = subprocess.run(["git", "log", "-1", "--format=%at", "--", str(pfad)],
                           cwd=pfad.parent, capture_output=True, text=True,
                           timeout=10)
        schmutzig = subprocess.run(["git", "status", "--porcelain", "--", str(pfad)],
                                   cwd=pfad.parent, capture_output=True,
                                   text=True, timeout=10)
        if schmutzig.stdout.strip():
            return None, "uncommittete Änderung"
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip()), None
    except Exception as e:
        return None, str(e)
    return None, "nicht in git"


def ts_zu_unix(ts):
    """'20260810T040512' → Unix-Zeit (lokal). Nur für den Integritätsvergleich."""
    import time
    try:
        return int(time.mktime(time.strptime(ts, "%Y%m%dT%H%M%S")))
    except Exception:
        return None


def pruefe(pfad, regel, nach_ts):
    print(f"\n{'=' * 68}")
    print(f"{regel.get('id', '?')} — {regel.get('frage', '')}")
    print(f"  Registrierung: {pfad.name}")
    print("=" * 68)

    g = regel.get("gueltige_nacht", {})
    arme = regel.get("arme", {})
    a_mit, a_ohne = arme.get("mit"), arme.get("ohne")
    ab = str(regel.get("serie_ab") or "")
    # naechte (Vorgabe) | tagesserie: N gepaarte Fits an EINEM Tag, je Paar
    # ein eigener Seed. Die Stichproben kommen dann aus der Seed-Ziehung
    # statt aus Naechten — der Seed-Sweep hat gemessen, dass die Naechte
    # ohnehin fast nur die Seed-Ziehung variieren.
    art = regel.get("serie_art", "naechte")
    quelle_soll = "tagesserie" if art == "tagesserie" else "nightly"

    n_soll = int(regel.get("naechte") or 0)
    gueltig, verworfen = [], []
    gesehene_tage = {}
    gesehene_seeds = {}
    for ts in sorted(nach_ts):
        if ab and ts[:len(ab)] < ab:
            continue
        zeilen = nach_ts[ts]
        # ⚠️ Still uebersprungen wird NUR, was nachweislich zur jeweils
        # anderen Serienart gehoert (Nacht-Zeilen in einer Tagesserie und
        # umgekehrt) — die laufen planmaessig parallel und sind kein
        # Defekt. ALLES andere bleibt laut: ein Handlauf oder eine Zeile
        # ohne quelle-Feld wuerde die Serie sonst still weiterzaehlen,
        # unter anderen Bedingungen gemessen. Der erste Entwurf dieser
        # Erweiterung hat genau das kaputt gemacht, und zwei bestehende
        # Tests haben ihn zurueckgewiesen.
        andere = {"tagesserie"} if quelle_soll == "nightly" else {"nightly"}
        m, o = zeilen.get(a_mit), zeilen.get(a_ohne)
        if not m or not o:
            # ⚠️ "Ein Arm fehlt" nur, wenn ueberhaupt einer der Arme DIESER
            # Regel da ist. Sonst flutet jede Tagesserie einer ANDEREN
            # Frage (fremde arch-Namen, gleiche quelle) dieses Audit fuer
            # immer mit verworfen-Zeilen — Laerm, der echte Luecken
            # unsichtbar macht.
            if not (m or o):
                continue
            vorhanden = {z.get("quelle") for z in zeilen.values()}
            if vorhanden and vorhanden <= andere:
                continue
            verworfen.append((ts, "ein Arm fehlt"))
            continue
        quellen = {z.get("quelle") for z in (m, o)}
        if quellen != {quelle_soll}:
            if quellen <= andere:
                continue
            wer = ("kein Nightly" if quelle_soll == "nightly"
                   else "keine Tagesserie")
            verworfen.append((ts, f"{wer} (quelle="
                             f"{'/'.join(sorted(str(q) for q in quellen))})"))
            continue
        if art == "tagesserie":
            # ⚠️ Gepaart heisst: BEIDE Arme mit demselben Seed. Die
            # baseline-Zeile der Nacht-Serie fittet fest mit Seed 0 —
            # genau deshalb gibt es diesen Serientyp ueberhaupt.
            if m.get("seed") != o.get("seed"):
                verworfen.append((ts, f"Arme nicht seed-gepaart "
                                      f"({m.get('seed')}/{o.get('seed')})"))
                continue
            # Ein Seed = eine Stichprobe. Derselbe Seed nochmal ist eine
            # Wiederholung — dieselbe Logik wie der Kalendertag unten.
            sd = m.get("seed")
            if sd in gesehene_seeds:
                verworfen.append((ts, f"Seed {sd} bereits gezählt "
                                      f"({gesehene_seeds[sd]})"))
                continue
            gesehene_seeds[sd] = ts
        else:
            # Eine Nacht pro Kalendertag. Zwei Laeufe am selben Tag sind
            # eine Wiederholung, keine zweite Stichprobe — bei ueber 99 %
            # Korpus-Ueberlappung erst recht.
            tag = ts[:8]
            if tag in gesehene_tage:
                verworfen.append((ts, f"zweiter Lauf am {tag} (gezählt: "
                                      f"{gesehene_tage[tag]})"))
                continue
        schlecht = [k for k, v in g.items()
                    if any(z.get(k) != v for z in (m, o))]
        if schlecht:
            verworfen.append((ts, "abweichend: " + ", ".join(schlecht)))
            continue
        if m.get("golden_median") is None or o.get("golden_median") is None:
            verworfen.append((ts, "kein golden_median"))
            continue
        if art != "tagesserie":
            gesehene_tage[ts[:8]] = ts
        gueltig.append((ts, round(m["golden_median"] - o["golden_median"], 4)))
        # ⚠️ Nach dem N-ten gueltigen Punkt ist die Serie VOLL — aufhoeren.
        # Zwei Gruende, beide keine Kosmetik: (1) spaetere gueltige Punkte
        # wuerden den Median einer bereits entschiedenen Serie retroaktiv
        # verschieben — ein Urteil, das sich mit jedem weiteren Lauf
        # aendert, ist keins. (2) Fragen TEILEN sich Arme (mlp32 steckt in
        # O2 UND der Kapazitaetsfrage) — ohne den Stopp erschiene jede
        # spaetere Serie hier als "ein Arm fehlt", fuer immer.
        if n_soll and len(gueltig) >= n_soll:
            break

    for ts, grund in verworfen:
        print(f"  verworfen {ts}: {grund} — Serie verlängert sich")

    einheit = "Paare" if art == "tagesserie" else "Nächte"
    if not gueltig:
        # "noch nicht begonnen" und "alle Naechte verworfen" sehen im
        # Ergebnis gleich aus, bedeuten aber Gegenteiliges: das eine ist
        # Warten, das andere ein kaputter Lauf, der niemandem auffaellt,
        # solange er als "noch offen" durchgeht.
        if verworfen:
            print(f"  Stand: 0/{n_soll} gültige {einheit} — ALLE "
                  f"{len(verworfen)} {einheit} verworfen. Das ist kein Warten, "
                  f"das ist ein Defekt: die Serie kommt so nie zustande.")
        else:
            print(f"  Stand: 0/{n_soll} gültige {einheit} — Serie hat noch "
                  f"nicht begonnen.")
        return True

    print(f"\n  {'Paar' if art == 'tagesserie' else 'Nacht':16s}  Δ (mit − ohne)")
    for ts, d in gueltig:
        print(f"  {ts:16s}  {d:+.4f}")

    deltas = [d for _, d in gueltig]
    med = statistics.median(deltas)
    neg = sum(1 for d in deltas if d < 0)
    b = regel.get("bedingungen", {})
    med_max = b.get("median_hoechstens")
    neg_min = b.get("negative_naechte_mindestens")

    print(f"\n  Median  {med:+.4f}   negativ  {neg}/{len(deltas)}")

    if len(deltas) < n_soll:
        print(f"\n  → NOCH OFFEN: {len(deltas)}/{n_soll} gültige {einheit}. "
              f"Zwischenstände sind KEIN Ergebnis.")
        return True

    c1 = med_max is None or med <= med_max
    c2 = neg_min is None or neg >= neg_min
    print(f"\n  Bedingung 1  Median ≤ {med_max}:            "
          f"{'erfüllt' if c1 else 'NICHT erfüllt'}  ({med:+.4f})")
    print(f"  Bedingung 2  ≥ {neg_min} von {len(deltas)} negativ:      "
          f"{'erfüllt' if c2 else 'NICHT erfüllt'}  ({neg})")

    if c1 and c2:
        print("\n  → REGEL ERFUELLT. Die vorab festgelegte Konsequenz gilt.")
    else:
        print("\n  → REGEL NICHT ERFUELLT. Die vorab festgelegte Konsequenz "
              "für diesen Ausgang gilt — nicht die Geschichte, die sich zu "
              "den Zahlen erzählen lässt.")
    return c1 and c2


def pruefe_integritaet(pfad, nach_ts, regel):
    """Wurde die Regel angefasst, nachdem die Serie begonnen hatte?

    ⚠️ Geankert wird auf der ersten Zeile, die zu DIESER Regel gehoert
    (Quelle + Arme) — nicht auf der ersten Zeile nach serie_ab ueberhaupt.
    Sonst schlaegt der Pruefer an, weil eine FREMDE Serie am selben Tag
    frueher lief: am 2026-08-12 markierte er die O2-Registrierung als
    nachtraeglich geaendert, weil die O1-Nightly-Zeilen von 03:30 vor dem
    16:31-Nachtrag lagen — O2s eigene Daten begannen erst 17:21. Ein
    Integritaetspruefer mit Fehlalarmen wird ignoriert, und dann uebersieht
    man den echten.
    """
    ab = str(regel.get("serie_ab") or "")
    art = regel.get("serie_art", "naechte")
    quelle_soll = "tagesserie" if art == "tagesserie" else "nightly"
    arme = set((regel.get("arme") or {}).values())
    erste = None
    for ts in sorted(nach_ts):
        if ab and ts[:len(ab)] < ab:
            continue
        if any(z.get("quelle") == quelle_soll and z.get("arch") in arme
               for z in nach_ts[ts].values()):
            erste = ts
            break
    if erste is None:
        return True
    stand, grund = regel_zuletzt_geaendert(pfad)
    if stand is None:
        print(f"\n  ⚠ INTEGRITAET: {pfad.name} — {grund}. Eine Regel, die "
              f"nicht festgeschrieben ist, ist keine Vorab-Registrierung.")
        return False
    # Tagesserien-ts tragen ein p-Suffix (…T172134p00) — fuer den
    # Zeitvergleich zaehlt die Basis.
    erste_unix = ts_zu_unix(erste.split("p")[0])
    if erste_unix and stand > erste_unix:
        print(f"\n  ⚠ INTEGRITAET: {pfad.name} wurde zuletzt NACH der ersten "
              f"gezählten Nacht ({erste}) geändert. Regel und Ergebnis sind "
              f"nicht mehr unabhängig — prüfen, was sich geändert hat:\n"
              f"     git log -p -- {pfad}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archiv", type=Path, default=ARCHIV)
    ap.add_argument("--docs", type=Path, default=DOCS)
    args = ap.parse_args()

    regeln = regeln_laden(args.docs)
    if not regeln:
        print("Keine Registrierung mit ```regel-Block gefunden.")
        return 0
    nach_ts = naechte_laden(args.archiv)
    ok = True
    for pfad, regel in regeln:
        ok &= pruefe(pfad, regel, nach_ts)
        ok &= pruefe_integritaet(pfad, nach_ts, regel)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
