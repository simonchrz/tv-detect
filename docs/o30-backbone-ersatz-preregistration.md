# O30 — Ersetzt SigLIP den alten Backbone, und tragen mehr Komponenten? (Vorab-Registrierung)

**Geschrieben 2026-09-30, vor dem ersten Behandlungs-Datenpunkt.** Bauart wie
O26–O29: offline, gleiche Zeilen, gleiche Seeds, gleiche Architektur. Skript:
`scripts/o30-backbone-ersatz.py`.

## Woher die Frage kommt

Seit 29.09. läuft MLP7 (Backbone 1280 + Logo + Audio + 3 OCR + SigLIP-64 + da).
Die Nachmessung vom 30.09. zeigt: die Voll/Halb-Lücke des Backbones zwischen
Training und Detect ist mit SigLIP praktisch zu (r 0.994), und im Fehlerbudget
steht das NN allein bei 0.939 statt 0.650. Der Backbone kostet 45–65 % der
Detect-Laufzeit und ist die einzige Spalte, die in Training (Python, voll) und
Detect (Go, halb) verschieden entsteht. Zwei Fragen, ein Lauf:

* **Ersatz:** Kommt der Kopf ohne den Backbone genauso weit? Dann fällt der
  Backbone samt seiner Lücke weg.
* **Breite:** O28 legte 64 Komponenten fest, bei ~200 Aufnahmen mit Spur. Jetzt
  sind es mehr; tragen 128 oder 256 messbar?

## Die Arme

| Arm | Spalten |
|---|---|
| K | Produktionszustand: Backbone + Logo (test halb) + Audio + OCR + SigLIP-64 + da (1350) |
| A | ohne Backbone: Logo + Audio + OCR + SigLIP-64 + da (70) |
| B | K mit SigLIP-128 |
| C | K mit SigLIP-256 |
| D | ohne Backbone, SigLIP-256 |

PCA jeweils NUR aus train-Zeilen mit Spur, wie O28.

## Rauschen, VOR der Behandlung gemessen

K, 5 Seeds, 23 menschlich gelabelte test-Aufnahmen: F1 0.9603 / 0.9578 /
0.9640 / 0.9621 / 0.9641, **Median 0.9621, sd 0.0027**
(`~/Library/Logs/o30-vormessung.log`; 686 train-Aufnahmen, 206 mit Spur).

## Regel

```regel
{
  "id": "O30",
  "frage": "Kommt der Kopf ohne den alten Backbone (nur SigLIP + Logo/Audio/OCR) genauso weit, und tragen 128/256 SigLIP-Komponenten mehr als 64?",
  "art": "offline-kopf-ab",
  "nicht_in_serienabschluss": true,
  "metrik": "F1 auf den MENSCHLICH gelabelten test-Aufnahmen mit halber Logo-Spalte UND SigLIP-Spur, geglaettet 10s",
  "arme": {"K": "Produktionszustand MLP7 (Backbone + Logo + Audio + OCR + SigLIP-64)", "A": "ohne Backbone, SigLIP-64", "B": "K mit SigLIP-128", "C": "K mit SigLIP-256", "D": "ohne Backbone, SigLIP-256"},
  "paarung": "gleicher Seed, gleiche Zeilen, gleiche Architektur",
  "seeds": 5,
  "rauschen_sd_kontrollarm": 0.0027,
  "bedingungen": {
    "median_delta_f1_mindestens": 0.005,
    "positive_seeds_mindestens": 4
  },
  "teilregeln": ["R1 (Ersatz, Nicht-Unterlegenheit): Median(A-K) >= -0.005 UND hoechstens 1 Seed unter -0.010", "R2 (Breite): Median(besserer(B,C) - K) >= +0.005 UND >= 4/5 positiv", "R3 (Ersatz breit): wie R1 fuer D"],
  "konsequenz_bei_erfuellt": "R1 oder R3 erfuellt: Produktionsweg OHNE Backbone VORSCHLAGEN (neues Kopf-Format, Go-Pipeline ohne Backbone-Decode, Paritaetstest; L5-OK). R2 erfuellt: n_siglip-Bump vorschlagen (Kopf-Format, Go-Lader). Nichts bauen ohne OK.",
  "konsequenz_bei_verfehlt": "Backbone bleibt, 64 Komponenten bleiben. Naechste Frage ist dann der Fenster-Kopf (O31), nicht die Eingaben."
}
```

## Warum diese Schwellen

±0.005 ≈ 2 sd des Kontrollarms. Nicht-Unterlegenheit für den Ersatz ist die
richtige Frage: der Gewinn wäre Einfachheit und Laufzeit, nicht F1. Für die
Breite reicht ein kleiner, aber sauberer Gewinn, weil der Umbau klein ist.

## Was diese Frage NICHT beantwortet

* Nicht den Live-Detect: der hat keine SigLIP-Spur; ohne Backbone hätte er
  fast nichts mehr. Ein Backbone-Wegfall bräuchte dort einen eigenen Weg.
* Nicht die Laufzeit des Detects: die wird bei einem Bau gemessen, nicht hier.
* Nicht die ~70 % des Korpus ohne Spur: Arm A/D haben dort nur Logo, Audio,
  OCR. Der Kopf lernt das mit; im Detect gibt es die Spur immer.
