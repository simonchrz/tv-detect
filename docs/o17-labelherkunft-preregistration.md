# O17 — Zählt das Training die richtigen Labels als menschlich?
(Vorab-Registrierung)

**Geschrieben 2026-09-06, bevor ein einziger Fit mit der korrigierten Regel
gerechnet wurde.** Bedingung, Armzahl und Schwelle stehen fest, bevor die
erste Zahl existiert (R4).

## Woher die Frage kommt

`train-head.py` bildet `which` aus der **blossen Existenz** von
`ads_user.json` (Z. ~3071):

```python
which = ("merged" if user.exists() and auto.exists()
         else "user" if user.exists() else "auto" if auto.exists() else "")
...
has_user = which in ("user", "merged")
```

`autoConfirmApply` in `tv-recorder/autoconfirm.go` legt genau so eine Datei
an — mit der **Detektorausgabe unverändert darin**. Für das Training ist
eine maschinell bestätigte Aufnahme damit von einer menschlich geprüften
nicht zu unterscheiden.

Gemessen 2026-09-06 an den 234 lebenden Aufnahmen mit nicht-leerem
`ads_user.json`: **139 menschlich (59 %), 78 maschinell (33 %), 17
agentengeschrieben (7 %)**. Im `train`-Eimer tragen heute **474** Aufnahmen
`has_user=True`; davon sind **115 nachweislich menschlich**, **65
nachweislich nicht** (49 maschinell + 16 Agent) und **294 tot und damit
nicht mehr entscheidbar**.

## Warum das mehr ist als ein Gewicht

`has_user` steuert vier Dinge, und alle vier sind auf „ein Mensch hat das
geprüft" gebaut:

| Stelle | Wirkung bei `has_user=True` |
|---|---|
| Z. 4407 | `--user-weight 2.0` — doppelter Trainingseinfluss |
| Z. 4358 | **Ausnahme vom Hygiene-Veto** — Labels, die dem Champion widersprechen, werden nicht aussortiert |
| Z. 6607 | **Reviewed-Regression-Veto** — ein Einbruch dort blockiert den Deploy |
| Z. 1713 | Ausnahme vom GT-Ausreißer-Wächter — kaputte Labels werden nie gemeldet |

Zusammen heisst das: die Ausgabe des vorherigen Champions wird doppelt
gewichtet, vor der Bereinigung geschützt, darf den Deploy blockieren und
wird nie als verdächtig gemeldet. Das ist derselbe Kreis wie in
`train_hygiene_champion_veto_echo_chamber`, eine Ebene tiefer — und er
steht neben O3, wo der Golden-Boden seit 08-09 über dem Champion liegt und
nichts mehr durchkommt.

## Die Frage

Ändert es das Modell messbar, wenn `has_user` nur noch dort gilt, wo ein
Mensch nachweisbar war?

## Arme

Nur **Arm 1** ist hiermit registriert. Arm 2 wird erst registriert, wenn
Arm 1 entschieden ist — sonst stehen zwei Änderungen in einer Zahl.

* **`ist`** — heutige Regel, unverändert.
* **`belegt`** — `has_user` nur, wenn `ads_user.json` **keinen**
  Auto-Marker trägt (`auto_confirmed_at`, `auto_confirmed_via_fingerprint`)
  und `reviewed_by` nicht in `NICHT_MENSCH` steht. **Aufnahmen ohne lesbare
  Quelle (archiv-injiziert) behalten `has_user` wie heute** — sie sind
  nicht entscheidbar, und sie mit hineinzuziehen wäre eine zweite Änderung.

**Nachtrag 2026-09-06, vor dem ersten Serienpaar — Reichweite präzisiert.**
Der erste Probelauf meldete **92** statt der oben genannten 65. Die 65 sind
train-only; der Schalter wirkt aber **korpusweit, vor dem Split**:

| Eimer | verlieren `has_user` |
|---|---|
| train | 65 |
| versiegelt | 17 |
| test | 13 |
| **Summe** | **95** (der Lauf meldet 92; drei Aufnahmen verwirft der Korpus ohnehin) |

Dass auch test und versiegelt betroffen sind, ist **gewollt und wichtiger
als der train-Anteil**: dort steuert `has_user` den GT-Ausreißer-Wächter
und das Reviewed-Regression-Veto. Ein maschinelles Label soll weder einen
Deploy blockieren noch von der Verdachtsliste ausgenommen sein.

⚠️ Geändert wurde hier eine **Beschreibung**, nicht die Entscheidungsregel.
Der ```regel-Block (Schwelle +0.010, 4 von 5) steht unverändert, und die
Serie hatte zu diesem Zeitpunkt **null** gültige Paare — es gibt nichts,
was diese Korrektur begünstigen könnte.

Arm 2 (später, eigene Registrierung): auch die 294 nicht entscheidbaren
verlieren das Privileg.

## Bedingung

```regel
{
  "id": "O17",
  "frage": "Zaehlt das Training die richtigen Labels als menschlich?",
  "serie_art": "tagesserie",
  "serie_ab": "20260906",
  "naechte": 5,
  "arme": {"mit": "mlp32-cwtmpwm-belegt", "ohne": "mlp32-cwtmpwm-ist"},
  "delta": "belegt minus ist, auf golden_median, beide Arme gleicher Seed",
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

**Warum 0.010 und warum in dieser Richtung.** Der Rauschboden (§2) ist
0.008 Std / 0.023 Spanne über 5 Seeds auf dem Golden-Satz. Eine Schwelle
unter 0.010 wäre nach R2 kein Ergebnis. Die Richtung ist **positiv**: die
Hypothese lautet, dass die Korrektur das Modell **verbessert**, weil sie
65 Aufnahmen mit Modell-Labeln aus der doppelten Gewichtung und aus dem
Hygiene-Schutz nimmt.

**Was ich erwarte — und warum ich es trotzdem messe.** Ehrlich: ich
erwarte, dass die Bedingung **verfehlt** wird. 65 von 474 sind 14 % der
doppelt gewichteten Aufnahmen, und der Golden-Satz ist ein grober Maßstab
für einen so lokalen Eingriff. Der eigentliche Schaden dieser vier
Ausnahmen liegt vermutlich nicht im Fit, sondern im **Veto** — und das
misst der Golden-Median gar nicht.

**Konsequenz bei Verfehlen (jetzt festgelegt, nicht hinterher):** Die
Korrektur wird trotzdem **eingebaut**, aber als Hygiene, nicht als
Verbesserung — mit der ausdrücklichen Feststellung, dass sie im Golden-Wert
keinen belegbaren Nutzen hat. Begründung: `has_user` behauptet etwas über
die Welt („hier war ein Mensch"), was nachweislich für 14 % falsch ist;
eine falsche Tatsachenbehauptung im Code ist auch dann zu korrigieren, wenn
die Metrik es nicht belohnt. Was NICHT passieren darf, ist die Korrektur
hinterher als Verbesserung zu erzählen.

**Konsequenz bei Erfüllen:** Einbau, und Arm 2 registrieren.

## Fallen im Aufbau — was ich erwartet hatte und was wirklich da war

**Die erwartete Falle gab es nicht.** Ich hatte notiert, ein Armname mit
Zusatz fiele durch die Architektur-Prüfungen (`args.head_arch in (...)`)
und ergäbe still einen nackten Kopf. Das trifft auf den Tagesserie-Pfad
**nicht** zu: `_ts_arme` ist eine Registry (Name → Spaltenbauer,
Kopfbreite), ein unbekannter Arm wird laut abgelehnt. Beide O17-Arme sind
deshalb als Einträge mit demselben Bauer `(_arm_prod, 32)` registriert —
die Architektur ist per Konstruktion identisch.

**Die echte Falle war eine andere, und sie ist im Trainer dokumentiert.**
`mensch_belegt` entsteht im ERSTEN Korpus-Durchgang (Z. ~3001) und wird im
ZWEITEN gebraucht (~3580) — zwei getrennte Schleifen. Wer die Variable im
zweiten einfach liest, bekommt den Wert der **letzten** Aufnahme des
ersten, und zwar still. Genau das ist an derselben Stelle schon einmal
passiert; der Kommentar an `confirmed_show` hält es fest: *„this loop read
the pass-1 loop variables, which by now hold the LAST recording's values —
233 of 591 archive entries share the identical [22.0, 1281.0]"*. Beim
Einbau wäre es fast ein zweites Mal passiert. `mensch_belegt` reist
deshalb durch `rec_info`, und `test_label_herkunft.py` hält das per AST
fest (Mutationsprobe: Transport entfernt → zwei Tests schlagen an).

**Der Nightly darf sich nicht mitverändern.** Die neue Regel steht hinter
`--herkunft-belegt`, Vorgabe AUS = heutiges Verhalten. Ohne den Schalter
ist die einzige Änderung die Berechnung von `mensch_belegt` (seiteneffekt-
frei) — der Fit ist unberührt.

**Der Arm-Lauf braucht zwei Prozesse.** Die Gewichte entstehen einmal
oberhalb der Armschleife (`sw_train_parts`); `_build_train` schneidet nur
daraus. Ein Gewichtungs-Unterschied lässt sich in EINEM Prozess also nicht
zwischen zwei Armen darstellen. Gefahren wird über
`--tagesserie-nur-arm` mit gemeinsamem `--tagesserie-ts` und
`--tagesserie-seeds` — der im Trainer vorgesehene Parallelbetrieb.

## Was diese Frage NICHT beantwortet

Ob der Maßstab selbst zu reparieren ist. `massstab-audit.py` zeigt: golden
22 von 38 nachweislich menschlich, test 7 von 102, **versiegelt 0 von 37**.
Der versiegelte Satz — die Gegenprobe gegen Selektionseffekte, auf der O5
ruht — enthält unter den lebenden Aufnahmen kein einziges menschliches
Label. Das ist eine eigene Frage und gehört nicht in diese.
