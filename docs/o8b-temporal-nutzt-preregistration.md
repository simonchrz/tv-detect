# O8b — Nützt die Temporal-Spalte jetzt (Wiederaufnahme)? (Vorab-Registrierung)

**Geschrieben 2026-08-19.** Spiegel-Frage zu
[`o8-temporal-preregistration.md`](o8-temporal-preregistration.md): jene
Serie fragte „**kostet** temporal mehr als es bringt?" und war ENTSCHIEDEN
(2026-08-13, nicht erfüllt: Median +0.0000, temporal war null). Diese hier
fragt das Gegenteil — „**nützt** temporal jetzt belegbar?" — weil sich das
Signal seither gedreht hat.

## ⚠️ R4-Offenlegung: die Wende ist SCHON gesehen

Ich habe die Schatten-Serie der letzten Nächte angesehen, BEVOR ich diese
Regel schreibe. Die gepaarte Differenz temporal−nackt:

```
  0813  +0.000   ┐
  0814  +0.004   │  = O8-Entscheidungsserie, Median +0.0000 (null)
  0815  -0.004   │
  0816  -0.003   ┘
  0817  +0.017   ┐
  0818  +0.008   │  = NEU, alle positiv, Median +0.0170
  0819  +0.022   ┘
```

Deshalb — genau wie bei O2, wo die zuerst gesehenen Nächte nicht zählten —
beginnt die neue Serie in der **Zukunft** (`serie_ab: 20260820`); die drei
Nächte 0817–0819 zählen NICHT. Und die Schwelle (0.010) ist wortgleich die
von O2/O8, NICHT an die gesehenen +0.017 angepasst.

## Anlass / Mechanismus-Vermutung

Der Umschwung fällt mit dem **Golden-Label-Audit vom 17.08.** zusammen (16
korrigierte Kanten, Label-Epoche geschnitten). Plausibel, nicht belegt:
sauberere Grenzen → die Temporal-Spalte (1-s-Delta über 31 s, s.
[[whisper_luecke_und_indikatorspalte]]) trägt jetzt echtes Signal statt in
verrauschten Kanten unterzugehen. Diese Serie prüft nur, OB der Beitrag da
ist — nicht warum.

## Regel

```regel
{
  "id": "O8b",
  "frage": "Nützt die Temporal-Spalte jetzt belegbar (≥0.010)?",
  "serie_art": "tagesserie",
  "serie_ab": "20260820",
  "naechte": 5,
  "arme": {"mit": "mlp32", "ohne": "mlp32-temporal"},
  "delta": "nackt minus temporal, beide Arme gleicher Seed. NEGATIV = temporal traegt bei",
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

Die Arme sind gegenüber O8 GETAUSCHT (`mit: mlp32`, `ohne: mlp32-temporal`),
damit „temporal nützt" als negatives Delta auf die vorhandene
`median_hoechstens`-Schwelle fällt — dieselbe geprüfte Auswertung, nur die
Frage gespiegelt.

## Was die Ausgänge bedeuten

**Erfüllt** = temporal trägt belegbar ≥0.010 bei. Konsequenz: der
Produktionskopf ist derzeit der nackte `mlp32` (Simons Entscheid 14.08.,
weil damals keine Spalte belegten Nutzen hatte). Hat temporal jetzt
belegten Nutzen, kippt die Abwägung Einfachheit-gegen-Nutzen für genau
diese eine Spalte — dann als **L5-Änderung** (`mlp32` → `mlp32-temporal`,
Header-Vertrag + `internal/signals/nn.go` in Tandem, Paritäts-Fixture)
Simon vorlegen, NICHT automatisch deployen.

**Nicht erfüllt** = die Wende war ein Drei-Nächte-Rauschen. Der nackte Kopf
bleibt, und O8 ist damit zum zweiten Mal bestätigt.
