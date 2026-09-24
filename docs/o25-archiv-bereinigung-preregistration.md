# O25 — Schadet es, die über den Defekt archivierten Aufnahmen aus dem Training zu nehmen? (Vorab-Registrierung)

**Geschrieben 2026-09-24, vor dem ersten Datenpunkt.** Bauart wie
[`o24-cluster-anker-preregistration.md`](o24-cluster-anker-preregistration.md):
Tagesserie, zwei Prozesse über `com.user.tv-tagesserie`, gemeinsamer
Stichtag, gemeinsame Seeds. Die Arme unterscheiden sich NUR in
`--archiv-ausschluss`. Gebaut als **Schadensprüfung**, nicht als
Verbesserungsfrage — dieselbe Haltung wie O17: eine Korrektur, die eine
verletzte Regel wiederherstellt, wird eingebaut, auch wenn die Metrik sie
nicht belohnt; die Serie soll nur verhindern, dass sie still etwas kaputt
macht.

## Woher die Frage kommt

`train-head.py` las `cluster_anchored.json` von 2026-05-03 bis 2026-09-23
über einen `rec_dir`-Restwert (die letzte Snapshot-Aufnahme). Die
Archiv-Regel „nur user / merged / auto-confirm **oder mit Ankern**“ war
dadurch für jede Live-Aufnahme wahr. Gemessen 2026-09-24:

* 163 Archiv-Einträge tragen ein rohes Detektor-Label (`which=auto`).
* **73** davon haben eigene Anker in einer Familie ≥3 — regelkonform.
  Sie stehen NICHT zur Frage (die Regel „auto + Anker“ ist eine eigene,
  spätere Frage).
* **90** haben keine eigenen Anker (80 gar keine Fingerprints) und kamen
  nur über den Defekt hinein. Liste mit Herkunft:
  [`archiv-ausschluss-o25.json`](archiv-ausschluss-o25.json).
* Die 90 liegen alle in train (89 + 1 ohne Ledger-Eintrag), keiner im
  Golden-, test- oder versiegelten Satz — der Maßstab ist nicht betroffen.
  In der Nacht 2026-09-24 trugen sie **11.4 % der Trainings-Frames,
  4.3 % des Gewichts**.
* 2 der 90 sind noch live; sie trainieren über den Live-Durchgang mit
  ihrem aktuellen Detektor-Label wie jede unbestätigte Aufnahme. Der
  Schalter betrifft nur die Einspeisung aus dem Archiv.

## Die Arme

| Arm | Name | Schalter |
|---|---|---|
| mit | `mlp32-bereinigt` | `--archiv-ausschluss docs/archiv-ausschluss-o25.json` |
| ohne | `mlp32-archivalt` | keiner (heutiges Verhalten) |

Beide nackt (`_ident`) wie das Nightly; `--cluster-anker aus` ist seit O24
Vorgabe und gilt in beiden Armen gleich.

## Regel

```regel
{
  "id": "O25",
  "frage": "Schadet es, die ueber den rec_dir-Defekt archivierten Aufnahmen aus dem Training zu nehmen?",
  "serie_art": "tagesserie",
  "serie_ab": "20260924",
  "naechte": 5,
  "arme": {"mit": "mlp32-bereinigt", "ohne": "mlp32-archivalt"},
  "delta": "bereinigt minus archivalt, auf golden_median, beide Arme gleicher Seed. NEGATIV = Bereinigung schadet",
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

## Was die Ausgänge bedeuten

**Erfüllt** = die Bereinigung schadet belegbar. Dann wird sie NICHT
eingebaut, und die Ursache wird untersucht: ein roher Detektor-Label, dessen
Entfernen belegbar schadet, trägt etwas, das kein anderer Teil des Korpus
trägt — das ist eine Frage für sich, keine Rechtfertigung für den Defekt.

**Nicht erfüllt** = kein belegter Schaden. Dann `--archiv-ausschluss` ins
Nightly, **als Hygiene, nicht als Verbesserung**. Auch ein positiver
Median wird nicht als Verbesserung erzählt — die Serie ist darauf nicht
angelegt. Die Archiv-Dateien selbst bleiben liegen (L6); ob sie gelöscht
werden, ist eine eigene Entscheidung.

## Ablauf

Tagsüber, nicht parallel zum Nightly (03:30), nach dem Commit dieser Datei:

```sh
launchctl kickstart gui/501/com.user.tv-tagesserie
```

(die Plist trägt die O25-Argumente; Log `~/Library/Logs/tv-tagesserie-o25.log`).
