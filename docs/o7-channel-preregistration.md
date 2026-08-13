# O7 — Kostet die Kanal-Spalte mehr als sie bringt? (Vorab-Registrierung)

**Geschrieben 2026-08-13, vor dem ersten Datenpunkt.** Bauart wie
[`o2-zusatzspalten-preregistration.md`](o2-zusatzspalten-preregistration.md)
(Tagesserie, seed-gepaart).

## ⚠️ Abweichung von der O2-Konsequenz — dokumentiert, nicht überspielt

Die O2-Registrierung sagt „O1 abwarten (14.08.), dann die verbleibenden
Spalten einzeln". Der Grund für das Warten war die **Whisper-Überlappung**
mit O1 — und die betrifft die Kanal-Spalte nicht: die Arme dieser Frage
enthalten kein Whisper, und O1s letzte Nacht wird von diesem Lauf nicht
berührt (eigene Quelle, eigene Arme). Der Wortlaut wird hier bewusst
verlassen, der Grund dahinter nicht.

Anlass: die seed-gepaarten Nightly-Deltas der letzten vier Nächte liegen
für Kanal−bare um null — eine Spalte, die über Serien positiv migriert
wurde, zeigt auf dem Golden-Satz keinen Beitrag mehr.

## Regel

```regel
{
  "id": "O7",
  "frage": "Kostet die Kanal-Spalte mehr als sie bringt?",
  "serie_art": "tagesserie",
  "serie_ab": "20260813",
  "naechte": 5,
  "arme": {"mit": "mlp32-channel", "ohne": "mlp32"},
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

## Was die Ausgänge bedeuten — und was nicht

**Erfüllt** = die Spalte schadet belegbar → sie fällt aus der
Ziel-Architektur.
**Nicht erfüllt** heißt NICHT „die Spalte bleibt zwingend". Ein
Null-Ergebnis liefert keinen belegten Beitrag — ob eine Null-Spalte in der
Architektur-Entscheidung nach O1 bleibt, wägt Einfachheit gegen belegten
Nutzen ab (dort entschieden, nicht hier nachverhandelt). Diese Serie
beantwortet nur: schadet sie messbar?
