# O2 — Schaden die Zusatzspalten in Summe? (Vorab-Registrierung)

**Geschrieben 2026-08-12, vor der ersten gezählten Nacht (13.08.).**
Muster und Begründung: [`o1-whisper-preregistration.md`](o1-whisper-preregistration.md),
Regeln R1–R5 in [`experiment-ledger.md`](experiment-ledger.md).

## Frage

Die Produktions-Architektur trägt fünf Zusatzspalten über dem nackten Kopf:
Kanal, Whisper, Temporal, Minute-Prior, Whisper-Maske. Jede wurde einzeln
positiv gemessen. Die Frage ist, ob sie **in Summe** noch tragen — oder ob
der schlankste Kopf inzwischen besser ist.

## ⚠️ Diese Registrierung ist teilweise nachträglich — und das steht hier

Die Schattenreihe fährt beide Arme seit dem 10.08. Ich habe die Zahlen also
**gesehen**, bevor ich diese Regel schreibe:

```
                                        0810   0811   0812
  mlp32-channel-whisper-temporal-mp-wm  0.906  0.907  0.910   (mit)
  mlp32                                 0.930  0.923  0.936   (ohne)
                                    Δ  -0.024 -0.016 -0.026
```

Das ist kein sauberer Vorab-Zustand. Zwei Konsequenzen, beide bindend:

1. **Diese drei Nächte zählen NICHT.** Die Serie beginnt am 13.08. und
   braucht fünf Nächte danach. Wer sie mitzählte, hätte die Frage mit den
   Daten beantwortet, die ihn auf die Frage gebracht haben.
2. **Die Schwelle wird NICHT an den beobachteten Effekt angepasst.** Sie ist
   wortgleich die von O1 (`median_hoechstens: -0.010`,
   `negative_naechte_mindestens: 4`) — geschrieben, bevor irgendwer O1s Daten
   sah. Eine Schwelle, die man nach dem Blick auf die Wirkung setzt, misst
   nur noch die eigene Erwartung.

Der beobachtete Effekt ist mit ~−0.022 gut doppelt so groß wie die Schwelle
und ein Vielfaches der Seed-Streuung (Std 0.006). Wenn er echt ist, wird die
Serie das zeigen. Wenn nicht, war er ein Artefakt der drei Nächte — und
genau dafür gibt es die Serie.

## Nachtrag 2026-08-12 (vor dem ersten Datenpunkt): Tagesserie statt Nächte

Die ursprüngliche Fassung (Serie 08-13 bis 08-17, nächtlich) trug einen
**Konstruktionsfehler**, der beim Bau des Tagesserien-Modus auffiel: die
`baseline`-Zeile der Schattenreihe — der „mit"-Arm dieser Frage — fittet
fest mit **Seed 0**, die Sonden mit dem Nacht-Seed. Die nächtliche Paarung
hätte also Spaltenwirkung und Seed-Differenz vermischt; bei einer
Seed-Streuung von Std 0.006 ist das kein Randfehler.

Deshalb, **bevor ein einziger Datenpunkt der Serie existiert**, Umstellung
auf eine Tagesserie: fünf gepaarte Fits auf dem heutigen Korpus, beide Arme
je Paar mit **demselben** Seed, Seeds über die Paare verschieden. Das ist
dieselbe Bauart wie die Nacht-Serie — der Seed-Sweep hat gemessen, dass die
Nächte ohnehin fast nur die Seed-Ziehung variieren (>99 %
Korpus-Überlappung). Was die Tagesserie NICHT misst, ist Korpus-Drift; die
Schwellen bleiben unverändert die vor jedem Datenblick festgelegten.

Präzedenz für einen Nachtrag vor Serienbeginn: der O1-Nachtrag vom
2026-08-09 (Seed-Konstruktionsfehler, ebenfalls vor der ersten Serien-Nacht
korrigiert und dokumentiert statt still ersetzt).

## Regel

```regel
{
  "id": "O2",
  "frage": "Schaden die Zusatzspalten in Summe?",
  "serie_art": "tagesserie",
  "serie_ab": "20260812",
  "naechte": 5,
  "arme": {"mit": "mlp32-channel-whisper-temporal-mp-wm", "ohne": "mlp32"},
  "delta": "mit minus ohne, auf golden_median, beide Arme gleicher Seed",
  "gueltige_nacht": {
    "set_hash": "c8727e8266a8",
    "decoder": "--decoder hsmm --hsmm-dur-w 15",
    "golden_n": 38
  },
  "bedingungen": {
    "median_hoechstens": -0.010,
    "negative_naechte_mindestens": 4
  }
}
```

## Was bei „erfüllt" folgt — und was ausdrücklich nicht

**Nicht:** die Spalten in einem Schritt entfernen. O2 sagt nur, dass die
Summe schadet; welche Spalte es ist, sagt es nicht. Kanal und Temporal sind
über Serien positiv gemessen worden, Whisper steht gerade in O1 zur
Disposition, und Minute-Prior wurde über acht Nächte als inert gemessen
(Δ ≈ −0.001).

**Sondern:** O1 abwarten (Urteil am 14.08.), dann die verbleibenden Spalten
einzeln in derselben Bauart prüfen — je eine Sonde, die sich um genau eine
Spalte unterscheidet. Die Reihenfolge richtet sich nach der Einzelmessung
der Schattenreihe, nicht nach Bequemlichkeit.

## ⚠️ Der Vorbehalt, der das Ergebnis begrenzen kann

Schatten-Varianten durchlaufen den **Refit auf allen Daten nicht**, die
Produktion schon. Und genau der ist die offene Spur aus O3: der
Produktionskopf landete am 11.08. auf Golden 0.900, *unter* dem gesamten
Seed-Pool (Minimum 0.906).

Ein Vorsprung in der Schattenreihe muss also nicht in die Produktion
übertragen. O2 beantwortet „ist der schlanke Kopf im Schatten besser?" —
nicht „wird die Produktion dadurch besser". Der zweite Schritt braucht einen
eigenen Produktionslauf gegen das Gate, und das ist ein Architekturwechsel
(Leitplanke L5: Header-Vertrag, Go-Inferenzpfad, Paritäts-Fixture).
