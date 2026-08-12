# O6 — Lässt der 32er-Kopf Leistung liegen? (Vorab-Registrierung)

**Geschrieben 2026-08-12 abends, vor dem ersten Datenpunkt.** Muster:
[`o1-whisper-preregistration.md`](o1-whisper-preregistration.md),
Tagesserien-Bauart wie [`o2-zusatzspalten-preregistration.md`](o2-zusatzspalten-preregistration.md).

## ⚠️ Das ist ein Friedhofs-Wiedergänger — und hier steht, warum er darf

MLP-64 wurde am 2026-07-12 nach einer 7-Nächte-Schattenserie als „Rauschen"
beerdigt (Code-Kommentar an der stillgelegten Sonde m_v5). Der Friedhof ist
bindend; eine Wiedervorlage braucht einen Grund, der die damalige Messung
entwertet — nicht bloß neue Hoffnung. Es gibt zwei:

1. **Die Serie war methodisch leer.** Bis 2026-08-09 fitteten alle
   Schattensonden jede Nacht mit `random_state=0`, bei >99 %
   Korpus-Überlappung. Sieben Nächte waren damit EINE Ziehung, siebenmal
   wiederholt — genau der Konstruktionsfehler, der am 08-09 gemessen und
   behoben wurde. Das Juli-Urteil sagt nichts über Kapazität; es sagt, dass
   dieselbe Münze siebenmal gleich fiel.
2. **Der Kontext hat gewechselt.** Damals wurde 64 auf der Kanal-Variante
   geprüft. Die relevante Architektur ist seit O2 der **nackte** Kopf —
   `mlp32` liegt 0.929–0.936, disjunkt über Produktion und Replika. Ob
   Kapazität auf DIESEM Kopf etwas hebt, wurde nie gemessen.

## Frage und Richtung

`mit` = `mlp32` (der Kandidat für die nächste Architektur), `ohne` = `mlp64`
(gleiche Eingaben, doppelte Kopfbreite). Δ = mit − ohne.

**Erfüllt** (Median ≤ −0.010 und ≥4/5 negativ) heißt: der 64er ist
verlässlich besser, der 32er lässt Leistung liegen → Kapazität wird Teil der
Architektur-Entscheidung nach O1.
**Nicht erfüllt** heißt: 32 bleibt, und MLP-64 geht zurück in den Friedhof —
diesmal mit einer Serie, die den Namen verdient, und einem Eintrag, der die
Wiedervorlage-Begründung gleich mitbeerdigt.

## Regel

```regel
{
  "id": "O6",
  "frage": "Laesst der 32er-Kopf Leistung liegen (Kapazitaet)?",
  "serie_art": "tagesserie",
  "serie_ab": "20260812",
  "naechte": 5,
  "arme": {"mit": "mlp32", "ohne": "mlp64"},
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

## Grenzen

* Ein Korpusstand, ein Abend — die Drift-Dimension fehlt wie bei jeder
  Tagesserie. Fällt das Urteil knapp, ist eine Bestätigungsnacht-Serie der
  nächste Schritt, nicht ein Nachverhandeln der Schwelle.
* `mlp64` kostet im Go-Inferenzpfad doppelte Kopf-Multiplikationen — bei
  dieser Kopfgröße irrelevant, aber ein „erfüllt" löst trotzdem KEINEN
  automatischen Wechsel aus (L5: Header-Vertrag, `nn.go`, Paritäts-Fixture).
