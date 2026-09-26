> **ABGESCHLOSSEN 2026-09-26 — R3 ERFÜLLT, R1 und R2 VERFEHLT.**
>
> | Teilregel | Median-Δ | positiv | Urteil |
> |---|---|---|---|
> | R1 M − K (SigLIP-Mittel je Aufnahme) | +0.0038 | 4/5 | verfehlt (Größe) |
> | R2 T − K (Sendungs-Kennung) | +0.0017 | 3/5 | verfehlt |
> | R3 S − M (sekundengenau gegen den besseren billigen Arm) | +0.0372 | 5/5 | **erfüllt** |
>
> K 0.9236 → S 0.9644 (sd 0.0025). Konsequenz laut Regel: SigLIP sekundengenau als
> Produktionsweg VORSCHLAGEN (zweiter Encoder, L5-Header-Bump mit ausdrücklichem OK).
> Ergebnis `~/.cache/tvd-train-archive/o29-ergebnis.json`, Log `~/Library/Logs/o29-lauf.log`.
>
> ⚠️ DEUTUNG per Gegenprobe auf K (`o28-gegenprobe.py --mit-ocr`, nach dem Urteil):
> nur Indikator −0.0011 (2/5), zeitlich gemischt **+0.0207** (5/5), sekundengenau +0.0404.
> Mit den OCR-Spalten trägt der Bezug zur Sekunde also rund die Hälfte (+0.020 ≈ 3 sd).
> Die andere Hälfte bleibt beim Mischen erhalten, lässt sich aber NICHT durch das Mittel je
> Aufnahme (R1) oder die Sendung (R2) ersetzen. Die O28-Deutung „zwei Drittel Kontext,
> billig zu haben“ ist damit widerlegt: was das Mischen überlebt, braucht trotzdem
> Einzelbilder. Vermutung, ungeprüft: Regularisierung durch bildnahes Rauschen oder eine
> Verteilungs-Information, die ein lineares PCA-Mittel über 200 Aufnahmen verliert. Für den
> Produktionsweg ändert das nichts: beide Hälften brauchen SigLIP je Sekunde.

# O29 — Trägt ein billiger Kontext-Träger den O28-Gewinn? (Vorab-Registrierung)

**Geschrieben 2026-09-26, vor dem ersten Behandlungs-Datenpunkt.** Bauart wie
O26–O28: offline, gleiche Zeilen, gleiche Seeds, gleiche Architektur. Skript:
`scripts/o29-kontext.py`.

## Woher die Frage kommt

O28 (SigLIP 2 als Zusatzblock) ist erfüllt: +0.0613, 5/5. Die Gegenprobe zeigte
aber, dass rund zwei Drittel davon ein zeitliches Mischen der SigLIP-Zeilen
überleben (+0.0407). Der Gewinn ist also überwiegend Kontext auf Aufnahme-Ebene
(welche Sendung, welcher Look). Ein zweiter Encoder in jedem Detect wäre dafür
teuer. Die Frage ist, ob ein billiger Träger denselben Kontext liefert.

Zusätzlich korrigiert O29 einen Rest-Konstruktionsfehler von O28: der
Produktionskopf MLP6 hat die drei OCR-Spalten, O28 trainierte ohne sie. Hier
hat JEDER Arm sie (Lehre O27: Kontrolle = Produktionszustand).

## Die Arme

| Arm | Inhalt |
|---|---|
| K | Produktionszustand: Archiv-Merkmale, test-Logo-Spalte halb, 3 OCR-Spalten |
| M | K + SigLIP-Mittel je Aufnahme (PCA-64 über train-Aufnahmen, je Aufnahme konstant) + `siglip_da` |
| T | K + Sendungs-Kennung: one-hot der Titel mit ≥ 3 train-Aufnahmen (Liste NUR aus train), Rest „sonstige“ |
| S | K + SigLIP sekundengenau (= O28-Versuch), Referenz |

## Rauschen, VOR der Behandlung gemessen

Kontrollarm K, 5 Seeds, 23 menschlich gelabelte test-Aufnahmen: F1 0.9236 /
0.9137 / 0.9296 / 0.9218 / 0.9260, **Median 0.9236, sd 0.0059**
(`~/Library/Logs/o29-vormessung.log`). Die OCR-Spalten allein heben also schon
von 0.8970 (O28-Kontrolle) auf 0.9236.

## Regel

Drei Teilfragen, alle gepaart je Seed, Schwelle je +0.012 (≈ 2 sd) und ≥ 4/5:

* **R1** M gegen K
* **R2** T gegen K
* **R3** S gegen den besseren der beiden billigen Arme (höherer Median aus M, T)

```regel
{
  "id": "O29",
  "frage": "Traegt ein billiger Kontext-Traeger (SigLIP-Mittel je Aufnahme oder Sendungs-Kennung) den O28-Gewinn, und lohnt SigLIP sekundengenau darueber hinaus?",
  "art": "offline-kopf-ab",
  "nicht_in_serienabschluss": true,
  "metrik": "F1 auf den MENSCHLICH gelabelten test-Aufnahmen mit halber Logo-Spalte UND SigLIP-Merkmalen, geglaettet 10s",
  "arme": {"K": "Produktionszustand inkl. 3 OCR-Spalten", "M": "K + SigLIP-Mittel je Aufnahme (PCA-64) + siglip_da", "T": "K + one-hot Titel (>=3 train-Aufnahmen) + sonstige", "S": "K + SigLIP sekundengenau (PCA-64) + siglip_da"},
  "paarung": "gleicher Seed, gleiche Zeilen, gleiche Architektur",
  "seeds": 5,
  "rauschen_sd_kontrollarm": 0.0059,
  "bedingungen": {
    "median_delta_f1_mindestens": 0.012,
    "positive_seeds_mindestens": 4
  },
  "teilregeln": ["R1: M - K", "R2: T - K", "R3: S - besserer(M, T)"],
  "konsequenz_bei_erfuellt": "R3 erfuellt: SigLIP sekundengenau als Produktionsweg VORSCHLAGEN (zweiter Encoder, Kosten messen, L5-Header-Bump mit ausdruecklichem OK). R3 verfehlt, R1 oder R2 erfuellt: den billigeren Traeger vorschlagen, T vor M, ausser M liegt im Median >= 0.012 ueber T. Nichts bauen ohne OK.",
  "konsequenz_bei_verfehlt": "Keine Teilregel erfuellt: die OCR-Spalten tragen den Kontext schon; SigLIP ist fuer jetzt erledigt."
}
```

## Was diese Frage NICHT beantwortet

* Nicht, wie T bei NEUEN Sendungen wirkt: die fallen auf „sonstige“. Test und
  train teilen Serien (Split per uuid-Hash); ein Erfolg von T gilt für
  bekannte Serien. Das Detect sähe für eine neue Sendung dasselbe wie heute.
* Nicht den Live-Detect: das SigLIP-Mittel braucht die ganze Aufnahme. Das
  Aufnahme-Detect läuft nach Aufnahmeende, dort ist das kein Problem.
* Nicht die ~70 % des Korpus ohne Quelle: M und S haben dort Nullen und
  `siglip_da=0`, T gilt überall.
