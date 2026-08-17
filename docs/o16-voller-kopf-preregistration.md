# O16 — Sieht der volle Kopf eine Flanke, wo der nackte keine hat?
(Vorab-Registrierung)

**Geschrieben 2026-08-17, bevor ein voller Kopf gegen den korrigierten
Golden-Satz gerechnet wurde.** Bedingungen und Vorhersage stehen fest, bevor
die erste Zahl existiert.

## Woher die Frage kommt

Aus §3ap, dem einzigen Befund des Tages, der eine Ursache benennt statt eines
Symptoms:

```
                      NN-Amplitude   Logo-Amplitude   (+-6 s um die WAHRE Kante)
gut      (<= 5 s) n=52     0.970          0.958
schlecht (> 10 s) n=17     0.628          0.980
```

**Wo das Modell weit danebenliegt, hat das NN keine Flanke.** Das Logo hat
dort eine, und sie liegt richtig — aber daraus wurde keine Regel, weil die
Bedingung zur Laufzeit nicht erkennbar ist und unbedingtes Ziehen aufs Logo
46 von 74 Kanten verschlechtert (§3ap).

Der Hebel liegt also nicht in der Auswahl, sondern darin, dass **das NN
mehr sieht**. Und genau das liegt fertig verdrahtet da: seit dem 2026-08-14
trainiert der Nachtlauf den **nackten** Kopf (`--head-arch mlp32`,
input 1282 = backbone + logo + audio). Die volle Architektur
(`mlp32-channel-whisper-temporal-mp-wm`, MLP4/5, input 1299 — 17 Spalten
mehr) ist gebaut, paritätsgetestet und nur durch eine Zeile abgeschaltet.

⚠️ Der Kommentar in `tv-train-head.sh` hält fest, dass die Umstellung **gegen
die Messlage** erfolgte: „bare 0.9332 vs prod 0.9372, Δ −0.0040". Dieses Δ
liegt weit innerhalb der Seed-Streuung (heute 0.0123–0.0229) und wurde gegen
die **unkorrigierten** Labels gemessen. Der Maßstab ist seit dem 17.08. ein
anderer — das ist der Grund, die Frage überhaupt neu zu stellen, und nicht
etwa ein Vorurteil zugunsten des vollen Kopfes.

## Der Eingriff — fixiert

`TVH_HEAD_ARCH_OVERRIDE` in `daemon/tv-train-head.sh` für **einen** Nachtlauf
leeren. Eine Zeile, kein Umbau, jederzeit zurückzudrehen. Sonst nichts:
kein anderer Decoder, keine Label-Änderung, kein Gate-Eingriff.

## Gemessen wird — und zwar NICHT primär der Golden-IoU

Der IoU ist für diese Frage zu stumpf: §3ao hat gezeigt, dass die Hälfte der
Kanten ohnehin auf 2 s sitzt und der Schaden im Schwanz steckt. Primär
gemessen wird deshalb der **Kantenfehler gegen den korrigierten Golden-Satz**,
mit demselben Werkzeug wie heute (Produktions-Decoder `hsmm --hsmm-dur-w 15`,
Vergleich gegen die am 17.08. bildgeprüften Kanten).

**Heutiger Stand, nackter Kopf, n=74 Kanten:**

```
Median 2.0 s     p75 10.0 s     p90 21.0 s     >10 s: 23 %
```

Gerechnet wird auf dem **Kandidaten**-Kopf aus `~/.cache/tv-train-head-out/`,
unabhängig davon, ob das Gate ihn deployt — sonst hinge die Messung an einer
Entscheidung, die eine andere Frage beantwortet.

**ERFÜLLT, wenn beides gilt:**

1. Der Anteil Kanten **über 10 s** sinkt um **≥ 5 Prozentpunkte** (also auf
   ≤ 18 %).
2. Der **Median** verschlechtert sich um **≤ 0.5 s** (also ≤ 2.5 s).

**Zusätzlich als Wächter, nicht als Bedingung:** der Golden-IoU darf nicht um
mehr als die Seed-Streuung dieses Laufs fallen. Fällt er weiter, wird das
Ergebnis unabhängig vom Kantenmaß als NICHT ERFÜLLT gewertet — ein Kopf, der
die Kanten schärft und die Blöcke verliert, ist kein Fortschritt.

Zusätzlich berichtet: NN-Flanken-Amplitude an den 17 heute schlechten Kanten
(die Mechanismus-Frage — sieht der volle Kopf dort überhaupt etwas?), und die
Aufteilung Starts gegen Enden.

## Vorhersage

**NICHT ERFÜLLT.** Begründung: die 17 Spalten sind Kontext (Kanal, Uhrzeit,
Whisper, zeitliche Nachbarschaft), nicht Bildinhalt. Wo das Backbone die
Flanke nicht sieht, ist unklar, warum ein Kanal-One-Hot sie erzeugen sollte.
Die Minute-Prior-Ablation war bereits Δ≈0 (Memory
`tv_detect_mlp4_minute_prior_migration`), und §3ap sagt, das fehlende Signal
ist ein **visuelles** (das Logo hat es, das Backbone nicht — und das Backbone
verwirft bei 224×224 genau solche Details, Memory `backbone_liest_keinen_text`).

Das ist die zweite Vorhersage in Folge, die „nein" lautet. Nach dem heutigen
Tag ist das der ehrlichere Ausgangspunkt.

## Was jeder Ausgang bedeutet — jetzt entschieden

**ERFÜLLT** → der volle Kopf bleibt an, und die Entscheidung vom 14.08. war
auf Rauschen gefällt. Der Golden-Boden wird dabei NICHT angefasst.

**NICHT ERFÜLLT** → der nackte Kopf bleibt, die Zeile wird zurückgesetzt, und
die Frage „woher bekommt das NN die fehlende Flanke" bleibt offen — dann
zeigt sie aber in Richtung **Bildinhalt** (Auflösung, Logo-Kanal ins
Backbone, OCR), nicht in Richtung Kontextspalten. Das wäre wieder eigens zu
registrieren.

⚠️ Kein Ausgang rechtfertigt, den Golden-Boden zu senken (L1), Labels
anzufassen (L2) oder die Bedingung nachträglich auf den IoU umzustellen, weil
das Kantenmaß nicht gefällt. Genau dieser Tausch ist der Grund, warum die
Kennzahl hier vorher benannt ist.
