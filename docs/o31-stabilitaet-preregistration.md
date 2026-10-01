# O31 — Bindet ein Stabilitaets-Ziel den neuen Kopf an den alten, ohne Qualitaet zu kosten? (Vorab-Registrierung)

**Geschrieben 2026-10-01, vor dem ersten Behandlungs-Datenpunkt.** Bauart wie
O26–O30: offline, gleiche Zeilen, gleiche Seeds, gleiche Architektur, Kontrolle
= Produktionszustand MLP7 (Backbone + Logo test halb + Audio + OCR + SigLIP-64 + da).
Skript: `scripts/o31-stabilitaet.py`.

## Woher die Frage kommt

Simon (2026-10-01): „wie können wir das hin und her verhindern?“ Jede Nacht wird
neu trainiert; der Kandidat tauscht einen Satz Fehler gegen einen anderen (26
Naechte in Folge ausgeliefert bei Median-Δ 0, letzte Nacht 18 von 144
test-Aufnahmen mit |ΔIoU| > 0.02, 4 mit > 0.1). Teil 1 (Gewinnpflicht im Gate,
`--nur-bei-gewinn 3`) ist seit 285879e scharf; er verhindert das Ausliefern,
nicht das Hin und Her im Kandidaten selbst. Diese Frage: Laesst sich der
Kandidat so trainieren, dass er dort, wo sich nichts geaendert hat, beim
Champion bleibt — und dort frei lernt, wo neue Labels sind?

## Behandlung

Zwei simulierte Naechte. Champion: trainiert OHNE die juengsten 10 % der
train-Aufnahmen (nach Aufnahmebeginn; 63 von 689), eigener Seed (1000+s).
Kandidat: alle train-Aufnahmen, Seed s. Weiches Ziel nur auf Aufnahmen, die der
Champion kannte: `(1−λ)·y + λ·p_champion` (ungeglaettete Champion-Ausgabe);
neue Aufnahmen behalten das harte Label. Klassengewichte nach dem HARTEN Label,
in allen Armen gleich (O20-Lehre).

| Arm | λ |
|---|---|
| K | 0 (heutiges Nightly) |
| S30 | 0.3 |
| S50 | 0.5 |

Gemessen auf den 23 menschlich gelabelten test-Aufnahmen mit halber Logo-Spalte
und SigLIP-Spur (wie O30), geglaettet (10 s), Schwelle 0.5:
* **kipp** = Anteil der Testframes, deren Entscheidung Kandidat ≠ Champion
* **F1** gegen die Labels

## Rauschen, VOR der Behandlung gemessen

K, 5 Seeds (`~/Library/Logs/o31-vormessung.log`): kipp 0.0083 / 0.0095 /
0.0060 / 0.0127 / 0.0085, **Median 0.0085, sd 0.0024**; F1 0.9637 / 0.9604 /
0.9686 / 0.9534 / 0.9643, **Median 0.9637, sd 0.0057**.

## Regel

Je Arm S, gepaart nach Seed gegen K:
* **R1 Stabilitaet:** Median(kipp_S / kipp_K) ≤ 0.70 UND kipp_S < kipp_K in ≥ 4 von 5 Seeds.
* **R2 Qualitaet:** Median(F1_S − F1_K) ≥ −0.003 UND kein Seed unter −0.010.

Ein Arm ist ERFUELLT, wenn R1 und R2 halten. Erfuellen beide, gilt S30
(kleineres λ bremst das Lernen weniger).

```regel
{"name": "O31 Stabilitaets-Ziel", "art": "offline", "skript": "scripts/o31-stabilitaet.py",
 "r1": "median(kipp_S/kipp_K) <= 0.70 und kipp_S < kipp_K in >= 4/5 Seeds",
 "r2": "median(F1_S - F1_K) >= -0.003 und min >= -0.010",
 "seeds": 5, "arme": {"K": 0.0, "S30": 0.3, "S50": 0.5}}
```

## Konsequenz

* **Erfuellt:** Produktionsweg vorschlagen (`--stabil-lambda` in train-head:
  Champion = Hygiene-Lehrer, p schon je train-Aufnahme berechnet; „kannte der
  Champion“ = Label-Abschluss und Aufnahmebeginn vor dem Champion-Deploy).
  Nightly-Umschaltung erst nach Simons OK (L5).
* **Verfehlt:** keine Trainingsaenderung; die Gewinnpflicht im Gate bleibt der Schutz.

## Grenzen (vorab benannt)

Offline Einzel-Seed-Koepfe; die Produktion liefert ein 3-Seed-Ensemble aus,
das schon weniger kippt. Frame-Entscheidungen statt Block-IoU nach HSMM — ein
Frame-Kipp ist notwendig, nicht hinreichend fuer einen Block-Kipp. Die
simulierte Nacht hat nur neue Aufnahmen, keine geaenderten Labels.
