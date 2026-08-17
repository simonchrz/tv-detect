#!/usr/bin/env python3
"""Golden-Labels gegen das Bildmaterial halten — Übersicht, ohne zu schreiben.

Liest die Urteile aus `~/.cache/tvd-agent-review/` und vergleicht die daraus
abgeleiteten Blöcke mit dem, was als Golden-Label in der Aufnahme steht.

⚠️ **Zwei Auflösungen, und die Verwechslung ist der ganze Punkt.** Ein
Grob-Urteil (45-s-Takt) meldet Blockstarts per Konstruktion im Mittel 22 s zu
spät — die wahre Grenze liegt irgendwo im Takt davor. Wer seine Abweichungen
als Kantenfehler liest, misst sein eigenes Raster (Ledger §3al, dort steht
auch, wie ich genau das einmal getan habe).

Deshalb meldet dieses Werkzeug aus GROBEN Urteilen **nur Abweichungen über
45 s** — die sind echt, alles darunter ist nicht unterscheidbar. Aus FEINEN
Urteilen (5-s-Takt um die grob gefundene Stelle) meldet es alles ab 5 s.

Es schreibt NICHTS. Der Golden-Satz ist der Maßstab; ihn zu ändern schneidet
über `label_hash` die Epoche und setzt die Latte neu auf. Das ist eine
Entscheidung, keine Messung — und wenn, dann für alle Funde auf einmal,
sonst zahlt jeder Einzelfund den vollen Preis.
"""
import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "ar", Path(__file__).resolve().parent / "agent-review.py")
_ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ar)

KURZ = {"sendungsinhalt": "S", "produktwerbung": "W", "programmvorschau": "T",
        "mitmachtafel": "M", "unklar": "?"}


def bilder_von(d):
    """Alle Bild-Urteile einer Aufnahme, über ALLE Runden hinweg.

    ⚠️ Nicht nur `urteil.json`. `--nachfassen` benennt das aktuelle Urteil in
    `urteil-r<n>.json` um, bevor es die nächste Runde anlegt — die erste
    Fassung las nur die eine Datei und meldete danach ein Dutzend Aufnahmen
    als „noch kein Urteil", obwohl deren Sichtung längst vorlag. `anwenden()`
    in agent-review.py sammelt aus demselben Grund über alle Runden.
    """
    je = {}
    for p in [d / "urteil.json"] + sorted(d.glob("urteil-r*.json")):
        if not p.is_file():
            continue
        try:
            roh = json.loads(p.read_text()).get("bilder") or []
        except Exception:
            continue
        for b in roh:
            try:
                je.setdefault(b["verzeichnis"], []).append(
                    (float(b["zeit"]), str(b["kategorie"])))
            except (KeyError, TypeError, ValueError):
                continue
    return je


def main():
    golden = _ar.golden_uuids()
    if golden is None:
        return 1
    funde, geprueft, offen, ohne_partner = [], 0, [], []
    for u in sorted(golden):
        d = _ar.ARBEIT / u
        lab = _ar.bloecke(_ar.SNAPSHOT / f"_rec_{u}" / "ads_user.json")
        if lab is None:
            continue
        # Feines Urteil schlägt grobes: kleinere Auflösung, echte Sekunden.
        auftrag_p = d / "auftrag.json"
        if not auftrag_p.is_file():
            offen.append((u, "nicht vorbereitet"))
            continue
        auftrag = json.loads(auftrag_p.read_text())
        je = bilder_von(d)
        if not je:
            offen.append((u, "vorbereitet, noch kein Urteil"))
            continue
        geprueft += 1
        if auftrag.get("art") == "grob":
            punkte = [p for v in je.values() for p in v]
            gef = _ar.bloecke_aus_grob(punkte)
            schwelle, quelle = _ar.GROB_TAKT_S, "grob"
            sicher = {(i, s) for i in range(len(gef)) for s in ("start", "ende")}
        else:
            # ⚠️ NUR bestimmte Kanten. Die erste Fassung hat bei einer
            # unbestimmten Kante den Wert aus `auftrag["bloecke"]` stehen
            # lassen — und der ist bei einem fein-nach-grob-Auftrag der
            # GROBE Wert. Gemeldet wurde er dann an der feinen 5-s-Schwelle,
            # also genau die Verwechslung der Auflösungen, gegen die dieses
            # Werkzeug geschrieben ist. Aufgefallen, weil kabel-eins mit
            # +54.4 s als „fein" erschien, obwohl die feine Ableitung dort
            # gar keine Kante liefert.
            gef = [list(x) for x in auftrag["bloecke"]]
            sicher, unbestimmt = set(), 0
            for k in auftrag["kanten"]:
                kante, _ = _ar.kante_aus_folge(je.get(k["verzeichnis"], []),
                                               k["seite"])
                if kante is None:
                    unbestimmt += 1
                    continue
                gef[k["block"]][0 if k["seite"] == "start" else 1] = kante
                sicher.add((k["block"], k["seite"]))
            if unbestimmt:
                offen.append((u, f"feine Runde, {unbestimmt} Kante(n) unbestimmt"))
            schwelle, quelle = _ar.SCHRITT_S, "fein"
        for i, g in enumerate(gef):
            if not lab:
                continue
            l = min(lab, key=lambda x: abs(x[0] - g[0]))
            # ⚠️ Ein gefundener Block ohne plausiblen Partner im Label ist
            # etwas ANDERES als eine verschobene Kante — er gehoert nicht als
            # riesige Abweichung gemeldet, sondern als eigener Fall. Ohne
            # diese Trennung erschien ein Artefakt-Block am Aufnahmeende als
            # Label-Fehler von +811 s.
            if min(abs(l[0] - g[0]), abs(l[1] - g[1])) > 180:
                ohne_partner.append((u, quelle, g))
                continue
            for j, seite in ((0, "start"), (1, "ende")):
                if (i, seite) not in sicher:
                    continue
                ab = g[j] - l[j]
                if abs(ab) > schwelle:
                    funde.append((u, quelle, seite, l[j], g[j], ab))
    print(f"{geprueft} Golden-Aufnahmen geprueft, {len(funde)} Kanten ueber "
          f"der jeweiligen Aufloesung:\n")
    for u, q, seite, lw, gw, ab in sorted(funde, key=lambda x: -abs(x[5])):
        print(f"  {u:36} {q:4} {seite:5} Label {lw:7.1f} -> Bilder {gw:7.1f}"
              f"  {ab:+7.1f}s")
    if ohne_partner:
        print(f"\n{len(ohne_partner)} gefundene(r) Block/Bloecke OHNE Partner im "
              f"Label (verpasster Block — oder ein Artefakt der Ableitung):")
        for u, q, g in ohne_partner:
            print(f"  {u:36} {q:4} {g[0]:7.1f}-{g[1]:7.1f}")
    if offen:
        print(f"\n{len(offen)} offen:")
        for u, warum in offen:
            print(f"  {u:36} {warum}")
    print("\n⚠️ Nichts geschrieben. Grobe Funde sind Kandidaten, keine Werte —"
          "\n   fuer die Sekunde braucht es `agent-review.py --vorbereiten --fein`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
