# O24 — Bringen die Cluster-Anker dem Training belegbar etwas? (Vorab-Registrierung)

**Geschrieben 2026-09-22, vor dem ersten Datenpunkt.** Bauart wie
[`o18-unentscheidbare-preregistration.md`](o18-unentscheidbare-preregistration.md):
Tagesserie, zwei Prozesse über `tv-tagesserie.sh`, gemeinsamer Stichtag,
gemeinsame Seeds. Die Arme unterscheiden sich NUR in `--cluster-anker`.

## Woher die Frage kommt

`cluster_anchored.json` markiert Spannen, deren Audio- und Bild-Abdruck
(Chromaprint + dHash) zu einer Spot-Familie mit mindestens drei Mitgliedern
gehört. `train-head.py` setzt dort das Label auf Werbung und gewichtet 1.5×.
Gemessen 2026-09-22:

1. **Die Anker sind kein unabhängiger Beleg.** `tv-spot-extract.py`
   fingerprintet nur INNERHALB der Label-Blöcke. Gegen die 139 menschlich
   gelabelten Aufnahmen im Snapshot liegen 99.0 % der Anker-Sekunden im
   Werbeblock — per Konstruktion. Ein Widerspruch entsteht nur, wenn das
   Label nach der Extraktion geändert wurde.
2. **Alle Widersprüche waren veraltete Fingerprints.** CSI: Miami
   `dvr-rtl-1781909700` (Golden-Satz): 21 Anker über reiner Sendung, an
   Bildern geprüft; Ursache `tv-recorder/ads.go` löschte Fingerprints nicht
   bei einem Save mit null Blöcken. South Park `dvr-nick-1782075600`: Anker
   hinter dem Ende der heute kürzeren Aufnahme. Beide 2026-09-22 behoben
   (ads.go + Trim-Pfad, 48 Zeilen gelöscht, Familien neu gebaut).
3. **Die Wirkung ist je Fit verschieden.** `y_train_parts` kopiert die
   Labels per bool-Maske, BEVOR die Anker-Schleife `r[4]` mutiert:

   | Fit | Label = Werbung | 1.5× Gewicht |
   |---|---|---|
   | Produktions-/Gate-Fit (`y_train`) | nein | ja |
   | All-Data-Refit (= `head.bin`) | ja | nein |
   | Tages-/Schattenserie (`_build_train`) | ja | ja |

   Das Gate misst also eine andere Behandlung, als ausgeliefert wird.
   Heute praktisch folgenlos (36 Frames in train von Sendung auf Werbung
   gekippt), aber der Mechanismus bleibt.

## Die Arme

| Arm | Name | Schalter |
|---|---|---|
| mit | `mlp32-anker` | `--cluster-anker alt` (heutiges Verhalten) |
| ohne | `mlp32-ohneanker` | `--cluster-anker aus` (keine Wirkung in keinem Fit) |

Beide **nackt** (`_ident`), wie das Nightly seit O2 ausliefert
(`TVH_HEAD_ARCH_OVERRIDE="--head-arch mlp32"`), nicht die Architektur, die
`tv-tagesserie.sh` per `--head-arch` vorgibt.

⚠️ **Was der mit-Arm misst:** die Serie baut ihre Matrix über
`_build_train` und bekommt damit Label UND Gewicht — die beabsichtigte
Behandlung laut Code-Kommentar, aber weder die des Gates noch die des
Refits. Gefragt ist deshalb: bringen die Anker, so wie sie gedacht sind,
etwas? Nicht: wie wirkt die heutige, inkonsistente Mischung.

`aus` liest die Anker weiter ein und lässt die Archiv-Entscheidung
(`or bool(cluster_anchored)`) unberührt — sonst unterschieden sich die Arme
im Korpus statt in der Behandlung (`test_cluster_anker.py`).

## Regel

```regel
{
  "id": "O24",
  "frage": "Bringen die Cluster-Anker dem Training belegbar etwas?",
  "serie_art": "tagesserie",
  "serie_ab": "20260923",
  "naechte": 5,
  "arme": {"mit": "mlp32-anker", "ohne": "mlp32-ohneanker"},
  "delta": "anker minus ohneanker, auf golden_median, beide Arme gleicher Seed. POSITIV = Anker tragen bei",
  "gueltige_nacht": {
    "set_hash": "c8727e8266a8",
    "decoder": "--decoder hsmm --hsmm-dur-w 15",
    "golden_n": 38
  },
  "bedingungen": {
    "median_mindestens": 0.010,
    "positive_naechte_mindestens": 4
  }
}
```

## Was die Ausgänge bedeuten

**Erfüllt** = die Anker tragen belegbar bei. Dann bleiben sie, aber die
Inkonsistenz muss weg: Label und Gewicht in ALLEN Fits vor der Kopie
anwenden. Das ist eine eigene Änderung mit Paritätsprüfung, nicht Teil
dieser Serie.

**Nicht erfüllt** = kein belegter Nutzen. Dann `--cluster-anker aus` ins
Nightly, weil drei Gründe gegen einen Mechanismus ohne belegten Nutzen
stehen: er liefert keine unabhängige Evidenz (Punkt 1), er ist der einzige
Weg, auf dem veraltete Fingerprints ein Label überschreiben (Punkt 2), und
er macht das Gate blind für das, was ausgeliefert wird (Punkt 3). Der Code
wird danach entfernt, nicht nur abgeschaltet. Das ist KEINE Aussage über die
Spot-Datenbank selbst — die bleibt für Auto-Confirm und `anker-mass.py`.

Eine Serie mit Vorzeichen, aber unter der Schwelle, zählt als nicht
erfüllt. Erwartet wird ein kleiner Effekt: die Anker liegen fast nur auf
Frames, die schon Werbung sind (1.5× Gewicht), und kippen kaum ein Label.

## Ablauf

Erst nach dem O23-Urteil (Loop am 2026-09-23 08:07), abgekoppelt:

```sh
nohup ~/src/tv-detect/daemon/tv-tagesserie.sh "mlp32-anker,mlp32-ohneanker" 5 \
  "--cluster-anker alt" "--cluster-anker aus" \
  > ~/Library/Logs/tv-tagesserie-o24-$(date +%Y%m%d).log 2>&1 &
```

Nicht parallel zum Nightly (03:30) starten: jeder Arm braucht rund 40 GB.

## Leitplanken

* Das Nightly bleibt bis zum Urteil auf `alt` (Vorgabe), bitgleich zu vorher.
* Keine Labels angefasst (L2). Gelöscht wurden nur veraltete Fingerprints
  zweier Aufnahmen; Sicherung auf dem Pi unter
  `/mnt/tv/hls/.spot-fingerprints.sqlite.vor-staleness-fix-20260922`.
* Die Skip-Press-Signale haben denselben Kopier-Fehler (Label nur im
  Refit). Sie betreffen 2 Frames und sind hier bewusst NICHT mitgemessen —
  zwei Änderungen in einem Arm messen zwei Dinge.
