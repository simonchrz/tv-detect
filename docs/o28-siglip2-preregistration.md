# O28 — Hebt ein text-fähiger Bild-Encoder (SigLIP 2) als Zusatzblock die Leistung? (Vorab-Registrierung)

**Geschrieben 2026-09-26, vor dem ersten Behandlungs-Datenpunkt.** Bauart wie
O26/O27: offline, gleiche Zeilen, gleiche Seeds, gleiche Architektur; die Arme
unterscheiden sich NUR im angehängten SigLIP-Block. Skript:
`scripts/o28-siglip.py`, Merkmale aus `scripts/o28-siglip-merkmale.py`.

## Woher die Frage kommt

Der Backbone sieht 224×224 und liest keinen eingeblendeten Text (Memory
`backbone_liest_keinen_text`); Trailer/Programmhinweise sind die größte
Fehlerklasse. O26 hat belegt, dass Text trägt (OCR-Spalten +0.031). SigLIP 2
(Google 2025, Variante NaFlex) verarbeitet Bilder in nativem Seitenverhältnis
und ist gerade im OCR-/Dokument-Bereich stark — ein Encoder, der Text
*sehen* kann, statt ihn in drei OCR-Flags zu verdichten.

Merkmale: `google/siglip2-base-patch16-naflex`, max_num_patches 576, je
Sekunde (ffmpeg `fps=1` wie im Training, danach Pixel-Seitenverhältnis-
Korrektur), 768 Dimensionen, `~/.cache/tvd-siglip2/<uuid>.npy`. Nur für
Aufnahmen mit Quelle (~30 % des Korpus).

## Die Arme (Lehre aus O27: Kontrolle = Produktionszustand)

| Arm | train | test |
|---|---|---|
| kontrolle | Archiv-Merkmale (1282) | Archiv-Merkmale, Logo-Spalte HALB (wie der Detect) |
| versuch | wie kontrolle + SigLIP-Block | wie kontrolle + SigLIP-Block |

SigLIP-Block = 64 Hauptkomponenten (PCA, NUR auf train-Zeilen mit Merkmalen
angepasst) + 1 Spalte `siglip_da`. Aufnahmen ohne Merkmale bekommen Nullen
und `siglip_da=0` (wie `spur_da` in O26). 64 statt 768: bei ~220 train-
Aufnahmen mit Merkmalen wäre der volle Block eine Überanpassungs-Einladung;
die Zahl steht hier vor der Messung fest und wird nicht nachgestellt.

## Grundgesamtheit

Primär zählen nur menschlich gelabelte test-Aufnahmen mit halber Logo-Spalte
UND SigLIP-Merkmalen — nur dort ist die Kontrolle wirklich der
Produktionszustand und der Versuch wirklich behandelt. Das Skript nennt die
Zahl vor dem ersten Fit; weicht sie von den 23 der O27-Vormessung ab, wird
das Kontrollarm-Rauschen im Lauf mitgemessen und berichtet, die Schwelle
bleibt trotzdem +0.012.

## Rauschen, VOR der Behandlung gemessen

Der Kontrollarm IST der Produktionszustand der O27-Vormessung (train voll /
test halb, 23 menschlich gelabelte test-Aufnahmen, 5 Seeds): F1 0.9054 /
0.8815 / 0.9129 / 0.8913 / 0.8970, **Median 0.8970, sd 0.0122**.

## Regel

```regel
{
  "id": "O28",
  "frage": "Hebt ein text-faehiger Bild-Encoder (SigLIP 2, PCA-64 + Indikator) als Zusatzblock die binaere Leistung?",
  "art": "offline-kopf-ab",
  "nicht_in_serienabschluss": true,
  "metrik": "F1 auf den MENSCHLICH gelabelten test-Aufnahmen, die SOWOHL die halbe Logo-Spalte (~/.cache/tvd-o27-logo-halb) ALS AUCH SigLIP-Merkmale haben, geglaettet 10s; Nebenwert: alle test-Aufnahmen mit beidem",
  "arme": {"kontrolle": "Produktionszustand: Archiv-Merkmale, test-Logo-Spalte halb", "versuch": "kontrolle + 64 SigLIP-2-Hauptkomponenten + siglip_da"},
  "paarung": "gleicher Seed, gleiche Zeilen, gleiche Architektur",
  "seeds": 5,
  "rauschen_sd_kontrollarm": 0.0122,
  "bedingungen": {
    "median_delta_f1_mindestens": 0.012,
    "positive_seeds_mindestens": 4
  },
  "konsequenz_bei_erfuellt": "Produktionsweg VORSCHLAGEN, nicht bauen: SigLIP-2 als zweiter Encoder im Detect (Kosten messen: ~1 min je Stunde Video auf dem Mac, dazu ONNX/CoreML-Export) und im Nightly fuer neue Aufnahmen; Kopf-Header-Bump (L5, ausdrueckliches OK). Vorher pruefen, ob der Gewinn die OCR-Spalten (O26) nur ersetzt oder ergaenzt.",
  "konsequenz_bei_verfehlt": "SigLIP 2 als Zusatzblock ist fuer jetzt erledigt. Der Text-Pfad bleibt die OCR-Spur."
}
```

## Warum diese Schwelle

+0.012 ≈ 1 sd des Kontrollarms. Ein Erfolg hätte echte Kosten (zweiter
Encoder in jedem Detect, Header-Migration); ein Effekt, der das nicht
deutlich trägt, gilt als verfehlt.

## Was diese Frage NICHT beantwortet

* Nicht, ob SigLIP den Backbone ERSETZEN könnte (Zusatzblock, nicht Tausch).
* Nicht das Zusammenspiel mit den OCR-Spalten des Produktionskopfs (MLP6):
  der O-Rahmen trainiert ohne sie. Bei Erfüllung ist genau das die nächste
  Frage.
* Nicht den Fall voller Abdeckung: nur ~30 % des Korpus hat eine Quelle.
  Ein Erfolg ist eine Untergrenze.
