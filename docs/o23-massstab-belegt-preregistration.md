# O23 — Misst der belegbare Maßstab etwas anderes als der ganze?
(Vorab-Registrierung)

**Geschrieben 2026-09-14, nachdem die erste Zahl existierte — und genau
deshalb steht sie hier oben, nicht versteckt:** am 14.09. lag der
Golden-Median über alle 38 Gepinnten bei **0.9599**, über die 22
nachweislich menschlich gelabelten bei **0.9895**, Abstand **−0.0297**.
Diese eine Zahl war der Anlass. Die Regel unten beurteilt ausdrücklich
**nicht** sie, sondern ob der Abstand über zehn Nächte *wandert* — dafür
existiert noch keine einzige Zahl, und das ist die Bedingung, unter der
eine nachträglich formulierte Frage noch ehrlich ist.

> ⚠️ **KORREKTUR 2026-09-15, bei 2 von 10 Nächten, vor jedem Urteil.**
> Die Zahlen oben (0.9599 / 0.9895 / −0.0297) gehören nicht zum Kopf vom
> 14.09., sondern zu dem vom **13.09.** Die Spur las fest die
> `champion`-Spalte aus `per-rec-iou.jsonl`, also den Kopf VOR dem Lauf;
> jede Zeile trug damit den Kopf der Vornacht. Aufgefallen, weil die Zeile
> vom 15.09. 0.9639 meldete, während der in derselben Nacht deployte Kopf
> 0.957 hatte. Richtig für den 14.09.: **0.9639 / 0.9909 / −0.0270**.
>
> Behoben in `produktionskopf()`: gemessen wird der Kopf, der NACH dem Lauf
> in Produktion ist (`candidate` bei Deploy, sonst `champion`). Beide
> bisherigen Trendzeilen sind aus derselben Quelle neu gerechnet und
> tragen ein Feld `korrigiert`; Sicherung `massstab-trend.jsonl.bak.20260915`.
>
> **Die Aussage ändert sich nicht:** über drei verschiedene Köpfe liegt der
> Abstand zwischen −0.027 und −0.033. Regel, Schwelle und Konsequenz
> bleiben unverändert — betroffen war nur, welcher Nacht eine Zahl
> zugeschrieben wird, und das vor der ersten beurteilten Zahl.

> ⚠️ **KORREKTUR 2026-09-17, bei 3 gültigen von 10 Nächten, vor jedem Urteil.**
> Die Spur schreibt eine Zeile je **Lauf**, die Auswertung zählte Zeilen —
> registriert sind **Nächte**. Drei Speicher-Messläufe am 16.09. standen so
> als drei Nächte drin („7/10“). Dazu lief die Nacht zum 17.09. wegen der
> launchd-Dateigrenze auf halbem Korpus; 8 Gepinnte fehlten, ihr Arm
> „alle“ war ein Median über 30 statt über 38 — nicht der registrierte Arm.
>
> Behoben in `naechte()` (`massstab-audit.py`, von `loop-status.py`
> mitbenutzt): je Kalendertag zählt die **erste Zeile mit allen 38
> Gepinnten**. Fällt der Nightly aus, zählt der erste vollständige
> Nachlauf desselben Tages — so waren der 15. und 16.09. schon behandelt.
> Die Datei bleibt unverändert, nur die Zählung filtert. Regel, Schwelle und
> Konsequenz bleiben unverändert. Ehrlich vermerkt: die Zeilenwerte waren
> sichtbar, aber alle gültigen Kandidaten eines Tages liegen zwischen
> −0.028 und −0.032; die Wahl „erste statt letzte“ verschiebt nichts, was
> die Schwelle von 0.010 berührt.

## Woher die Frage kommt

Jede registrierte Frage von O1 bis O18 wurde am Golden-Median entschieden.
Woraus dieser Satz besteht, hat bis heute niemand nachgesehen — das
Skript dafür (`scripts/massstab-audit.py`) lag seit dem 06.09. im Repo und
wurde **nie aufgerufen**.

Stand 14.09.:

| Eimer | n | Mensch | Maschine | unbekannt |
|---|---|---|---|---|
| golden | 38 | 22 | 3 | 13 |
| test | 113 | 7 | 24 | 82 |
| versiegelt | 44 | 0 | 22 | 22 |

„Unbekannt" heißt hier fast immer `which=merged` im Archiv — die
Verschmelzung aus Auto- und Nutzer-Labeln, die einen Menschen *abdeckt*,
aber nicht *belegt*. O17 hat gezeigt, dass `which` als Herkunftsnachweis
untauglich ist: `autoConfirmApply` legt dieselbe Datei mit der
Detektorausgabe darin an.

Die Richtung des Abstands überrascht. Die naheliegende Sorge war
„Gratispunkte" — ein Label, das die Modellausgabe IST, erzeugt Fehler
null. Gemessen ist das Gegenteil: die nicht belegten Mitglieder schneiden
**schlechter** ab und ziehen den Median um 0.03 nach unten. Die
plausibelste Lesart: sie wurden nie geprüft und sind häufiger schlicht
falsch, und das Modell wird dafür bestraft, ihnen zu widersprechen. Das
passt zur Label-Seite des Fehlerbudgets (919 s „Anker sagt Werbung, Label
Sendung").

## Die Frage

Ist der Abstand ein **konstanter Versatz** — beide Maßstäbe messen
dasselbe, einer nur strenger — oder **driftet** er? Im zweiten Fall
entfernt sich der Gate-Boden mit jeder Nacht weiter von dem, was je ein
Mensch bestätigt hat, und niemand merkt es.

## Warum die Antwort nicht offensichtlich ist

* **Für konstanten Versatz:** die Zusammensetzung des Golden-Satzes ist
  seit dem 27.07. eingefroren. Was sich ändert, ist nur der Kopf — und der
  trifft beide Teilmengen gleichzeitig.
* **Für Drift:** der Kopf wird jede Nacht auf einem Korpus trainiert, in
  dem der Anteil maschineller Labels wächst, seit niemand mehr reviewt. Ein
  Modell, das zunehmend auf Maschinenlabel passt, muss sich auf den
  belegten Mitgliedern anders entwickeln als auf den unbelegten. Der
  Golden-Satz selbst ist konstant, der Trainer daneben nicht.

## Arme

Dieselbe Nacht, derselbe Kopf, dieselbe `per-rec-iou`-Datei — nur zwei
Teilmengen derselben 38 Aufnahmen:

* **alle** — Median über alle 38 Gepinnten (die heutige Gate-Grundlage)
* **belegt** — Median über die Teilmenge mit nachweislich menschlichem
  Label (heute 22; wächst, wenn reviewt wird)

Es gibt keinen zweiten Trainingslauf und keinen zweiten Kopf. Diese Frage
kostet keine Rechenzeit.

## Bedingung

Über **10 aufeinanderfolgende Nächte** im `massstab-trend.jsonl`:

> Der Median des Abstands über die letzten 5 Nächte unterscheidet sich vom
> Median über die ersten 5 um **mindestens 0.010**.

**Woher die Schwelle kommt, vor der ersten Zahl.** Der Rauschboden des
Golden-Medians ist 0.008 Std über 5 Seeds bei n=38 (§2). Die belegte
Teilmenge hat n=22, ihr Median streut um Faktor √(38/22) ≈ 1.31 breiter,
also ~0.010. Der Median über fünf Nächte drückt das noch einmal um ~√5 —
die Schwelle liegt damit bewusst über dem, was Rauschen erklären kann.

## Vorhersage, vor der ersten Zahl

**Konstanter Versatz, also nicht erfüllt.** Der Satz ist eingefroren, und
zehn Nächte sind kurz gegen die Geschwindigkeit, mit der sich der
Korpus-Mix verschiebt. Wenn ich falsch liege, ist es das interessantere
Ergebnis — dann ist der Gate-Boden nachweislich unterwegs.

## Konsequenz bei Erfüllen

Der Maßstab wird umgestellt: der Gate-Boden rechnet künftig auf der
belegten Teilmenge. Das ist ein eigener, registrierter Schritt mit
Dual-Zeile im Trend, bis alle laufenden Serien umgestellt sind — eine
Umstellung ohne Übergang bräche die Vergleichbarkeit mit jeder früheren
Nacht (Leitplanke L1, `golden_boden()`: Kompositions-Konstanz).

## Konsequenz bei Verfehlen

Der volle Satz bleibt Gate-Grundlage. Der Abstand ist dann ein bekannter,
konstanter Versatz — er wird weiter jede Nacht berichtet, aber er
begründet keine Umstellung. **Kein** „fühlt sich trotzdem besser an".

## Was diese Frage NICHT beantwortet

**Welcher der beiden Maßstäbe RECHT hat.** Ein konstanter Versatz sagt
nur, dass die unbelegten Mitglieder systematisch schwerer sind — nicht, ob
sie falsch gelabelt sind. Das kann nur ein Review entscheiden, nicht eine
Statistik.

## Der Hebel, der hier nicht gemessen wird

Die belegte Teilmenge kann heute **nicht wachsen**: der Golden-Satz muss im
test-Eimer liegen (Leakage-Freiheit), und dessen menschlich gelabelte
Aufnahmen sind erschöpft — alle brauchbaren sind bereits drin
(`golden_v3_vorschlag.py`: Kandidaten = 0). 84 weitere test-Aufnahmen
haben kein VOD mehr und sind nie wieder reviewbar.

Was bleibt, sind **29 reviewbare test-Aufnahmen ohne menschliches
Review** — jede davon wird nach dem Review ein Golden-Kandidat:

| Kanal | n |
|---|---|
| prosieben | 8 |
| rtl | 6 |
| kabel-eins | 5 |
| vox | 5 |
| disney-channel | 2 |
| one-hd, rtlzwei, sixx | je 1 |

Das ist die einzige Stellschraube, die den Maßstab wirklich verbessert,
und sie ist menschliche Arbeit. 29 Reviews würden den belegten Kern von 22
auf bis zu 51 heben — und damit seine Streuung von 1.31× auf 0.86× des
heutigen Satzes drücken.
