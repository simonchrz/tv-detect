# Experiment-Ledger — offene Fragen, Entscheidungsregeln, Friedhof

**Zweck.** Dieses Repo verbessert sein Modell in einer nächtlichen Schleife.
Die Schleife darüber — welche Frage als nächstes drankommt, wann eine Serie
lang genug ist, was schon entschieden wurde — lief bisher nur in Köpfen und
Memories. Dieses Dokument ist ihr Gedächtnis.

**Vor jedem Vorschlag zuerst den Friedhof lesen.** Die Hälfte der
naheliegenden Ideen ist hier schon gestorben, oft nach mehreren Nächten
Messung. Eine Idee erneut vorzuschlagen kostet nicht nur Rechenzeit, sie
kostet Vertrauen in die Serie.

Verwandt: [`hsmm-holdout-preregistration.md`](hsmm-holdout-preregistration.md)
(das Muster für Vorab-Registrierungen), `docs/dataflow.md` im tv-receiver-Repo
(Invarianten des Gesamtsystems).

---

## 1. Entscheidungsregeln

**⚠️ Bruch in der Reihe am 2026-08-10.** Bis 08-09 lieferte der Nightly
immer den Seed-0-Fit aus, seither den mittleren von dreien. Damit misst der
Golden-Median seither **eine andere Größe** — wer 08-10 mit 08-09 vergleicht,
vergleicht zwei Auswahlregeln und nicht zwei Modelle. Das ist genau die
stille Neudefinition, gegen die R1 unten steht, und sie stammt aus einer
Änderung von mir. Seit 08-10 steht `seed_golden` in jeder Zeile; `["0"]` ist
der Wert nach der alten Regel, `loop-status.py` zeigt ihn neben dem neuen.
Der Wert für 08-10 selbst fehlt (die Spalte kam einen Lauf zu spät).

**R1 — Nur der Golden-Median ist über Nächte vergleichbar.** Der Testsatz
wechselt seine Zusammensetzung mit dem Korpus; der Golden-Satz (38 gepinnte
Aufnahmen, `set_hash c8727e8266a8`) nicht. Jede Zahl ohne `set_hash` und
Decoder daneben ist wertlos: der Satz hat schon einmal still seine
Zusammensetzung gewechselt, und der Wechsel `form` → `hsmm` hat jede Zahl
davor entwertet.

**R2 — Kein Ergebnis unterhalb des Rauschbodens.** Der Boden wird mit
`--seed-sweep N` gemessen (gleiche Daten, gleiche Architektur, anderer
Init-Seed). Was darunter liegt, ist kein Ergebnis, egal wie plausibel die
Geschichte dazu klingt. *Gemessen 2026-08-09: siehe §2.*

**R3 — Serie statt Einzelnacht.** Präzedenzfälle: temporal migriert nach 7
Nächten (6/7 positiv), Minute-Prior nach 3. Eine Einzelmessung entscheidet
nie. Beobachtet 2026-08-09: dieselbe Sondenzelle schwankte über drei Läufe um
0.028 — zwei zufällig gleiche Zahlen (0.946 an zwei Abenden) sahen aus wie
Reproduzierbarkeit und waren keine.

**R4 — Vorab-Registrierung vor Nacht 2.** Hypothese, Nächtezahl und Schwelle
werden aufgeschrieben, *bevor* die Serie sie beantworten kann. Das Repo hat
zweimal für die Umkehrung bezahlt (Boundary-Head +0.016 auf unsauberen Dumps
gegen −0.015 auf sauberen; erstes HMM-Ergebnis +0.242 gegen einen falsch
gelabelten Baseline-Wert).

**R5 — Das Gate entscheidet, nicht die Serie.** Eine Serie sagt, ob eine
Änderung in den Produktions-Fit darf. Ob der resultierende Kopf deployt wird,
entscheidet allein das Head-to-Head plus Golden-Boden.

## 2. Rauschboden

| gemessen | Verfahren | Std | Spanne |
|---|---|---|---|
| 2026-08-09 | 5 Seeds, Golden-Satz (38 Aufnahmen) | **0.008** | **0.023** (0.901–0.924) |
| 2026-08-09 | 3 Seeds, Testsatz (98 Aufnahmen) | — | **0.010** (0.913–0.923) |

Derselbe Zufall, an zweieinhalbmal so vielen Aufnahmen gemessen, ergibt
weniger als die halbe Streuung. Ein erheblicher Teil des Golden-Rauschens
kommt schlicht daher, dass 38 Aufnahmen für einen Median wenig sind — ein
größerer Golden-Satz wäre auch ein ruhigerer.

**Ab 2026-08-09 wird der Boden nicht mehr einmalig gemessen, sondern
mitgeschrieben:** der Nightly fittet drei Produktionsköpfe mit verschiedenen
Seeds und liefert den mittleren aus (`--prod-seeds 3`); `seed` und
`seed_spread` stehen in jeder `golden-trend.jsonl`-Zeile. Jede Nacht liefert
damit einen Datenpunkt dazu, wie viel von einer Golden-Differenz überhaupt
Signal sein kann.

Beobachtete Schwankung derselben Schattenzelle über drei Läufe mit fast
gleichem Korpus (08-08, 08-09 Nightly, 08-09 Handlauf): bis 0.028. Der
Sweep zeigt: der Löwenanteil davon ist **Fit-Zufall**, nicht Korpus-Drift.

**Drei Konsequenzen, die weiter reichen als O1:**

* **Der Golden-Verlauf ist flacher als das Rauschen.** Sechs Nächte spannen
  0.907–0.921 (0.014); ein anderer Seed auf identischen Daten spannt 0.023.
  Erzählungen über „langsamen Drift" brauchen mehr als diesen Verlauf.
* **Der Boden ist teilweise Seed-Glück.** 0.921 stammt aus einem
  `random_state=0`-Fit vom 08-07. Die Sperrklinke verlangt seither, dieses
  Glück zu schlagen — ein plausibler Mechanismus für O3. **Trotzdem nicht
  senken** (Leitplanke L1): die richtige Antwort ist ein robusterer
  Bodenwert, nicht ein niedrigerer.
* **Serien brauchen wechselnde Seeds.** Bei über 99 % Korpus-Überlappung
  zwischen zwei Nächten ist eine Serie mit festem Seed eine Messung, N-mal
  wiederholt. Seit 08-09 wechselt der Seed je Nacht und steht in jeder
  jsonl-Zeile.

## 3. Offene Fragen

### O1 — Kostet die Whisper-Spalte mehr als sie bringt?

*Status: registriert 2026-08-09, Serie 08-10 bis 08-14 (5 Nächte). Regel und
Schwellen stehen in [`o1-whisper-preregistration.md`](o1-whisper-preregistration.md)
— vor der ersten Serien-Nacht geschrieben.*

Drei unabhängige Beobachtungen zeigen in dieselbe Richtung:

* Golden-Verlauf seit der Ausrichtungs-Korrektur: 0.921 → 0.917 → 0.911.
  Vorher stand im Fit eine zeitversetzte, also faktisch zufällige Spalte —
  Rauschen lernt ein Netz zu ignorieren. Seit der Korrektur trägt sie ein
  echtes Signal bei, und der Trend zeigt nach unten.
* Die Sonde `channel + whisper` liegt in zwei von drei Läufen unter
  `channel` allein.
* Die isolierte Sonde (`ct+mp` gegen `cwt+mp`, Unterschied genau eine
  Spalte), Handlauf 2026-08-09: Test −0.032, Golden −0.025.

Dagegen: der Handlauf drehte gleichzeitig die Minute-Prior-Sonde von +0.002
auf −0.023 — ein gepaarter Vergleich derselben Bauart. Ein Lauf ist hier
ungefähr ±0.025 wert, also genau die Größenordnung des Effekts.

**Wenn die Serie hält:** Header-Bump auf MLP6 ohne Whisper und Maske
(1301 → 1299), Go-Seite in `internal/signals/nn.go` in Tandem, Paritäts-Fixture
erweitern. Nebeneffekt: `--nn-whisper-json` und die Kopplung des Inferenzpfads
an die Transkriptions-Pipeline fallen weg. Das ist ein Architekturwechsel →
Leitplanke L5.

### O2 — Schaden die Zusatzspalten in Summe?

*Status: **ENTSCHIEDEN 2026-08-12, REGEL ERFÜLLT** — als erste Frage per
Tagesserie am selben Tag. Median Δ **−0.0249**, 5/5 Paare negativ (Schwelle
−0.010 / ≥4), Paare seed-gleich auf identischem Korpus (577 Train-Aufnahmen).
Audit sauber, Regel committet 16:31, erster Datenpunkt 17:21. Die Zusatz-
spalten schaden IN SUMME. Konsequenz laut Registrierung: NICHT alle Spalten
in einem Schritt entfernen — O1 abwarten (14.08.), dann Einzelspalten-Fragen
(§3a Punkt 1) und die Refit-Frage (§3a Punkt 2), BEVOR ein Architektur-
wechsel ansteht (L5).*

Im Handlauf 2026-08-09 hatte die schlankste Variante überhaupt — `MLP-32`
ohne jede Zusatzspalte — den höchsten Golden-Median der Tabelle. Die
Schattenreihe hat das seither bestätigt:

```
                                        0810   0811   0812
  mlp32-channel-whisper-temporal-mp-wm  0.906  0.907  0.910   (Produktion)
  mlp32                                 0.930  0.923  0.936
                                    Δ  -0.024 -0.016 -0.026
```

Drei Nächte, ~−0.022 im Mittel, bei einer Seed-Streuung von Std 0.006. Das
ist das stärkste Signal der gesamten Tabelle — deutlich stärker als O1.

⚠️ **Meine Begründung, O2 hinter O1 zu stellen, war falsch.** Sie lautete:
„zwei Serien gleichzeitig auf demselben Rauschboden sind nicht
auseinanderzuhalten." Das gilt für zwei Änderungen an DEMSELBEN Kopf. Die
Sonden hier sind aber **gepaart innerhalb einer Nacht**, unterscheiden sich
um genau eine Sache, fassen die Produktion nicht an — und der Nightly fährt
ohnehin acht Varianten. O1 und O2 sind zwei unabhängige Differenzen aus
demselben Lauf. Die Serialisierung hat die Schleife grundlos verlangsamt.

⚠️ Die drei Nächte oben zählen für die Serie **nicht** — ich habe sie
gesehen, bevor ich die Regel schrieb. Serienbeginn ist der 13.08., und die
Schwelle ist wortgleich die von O1, damit sie nicht an den beobachteten
Effekt angepasst ist. Beides steht in der Registrierung.

## 3a. Warteschlange

Eine Frage ist **eingereiht**, sobald ihre Registrierung existiert und
`serie_ab` in der Zukunft liegt; das Audit führt sie dann als „Serie hat
noch nicht begonnen". Es braucht kein eigenes Queue-Format — die
Registrierung IST der Eintrag, und damit gilt R4 (Regel vor den Daten)
automatisch auch für alles, was noch wartet.

Was ansteht, in dieser Reihenfolge:

1. **Welche Spalte genau?** → Kanal und Temporal sind whisper-frei und
   laufen als **O7/O8** seit 2026-08-13 (ein Lauf, drei Arme, geteilter
   bare-Arm; Abweichung von der O2-Konsequenz in beiden Registrierungen
   dokumentiert). Whisper selbst ist O1; Maske hängt an dessen Urteil.
   Ursprünglicher Plan: erst wenn O1 und O2 entschieden sind. Je eine
   Sonde, die sich um genau eine Spalte unterscheidet. Kanal und temporal
   sind über Serien positiv gemessen, Minute-Prior über acht Nächte als
   inert (Δ ≈ −0.001) — die Reihenfolge richtet sich nach der
   Einzelmessung der Schattenreihe.
2. **Trägt der Schatten-Vorsprung in die Produktion? — BEANTWORTET
   2026-08-12 abends, ohne neuen Lauf.** Zwei Code-Funde und ein
   Daten-Kreuzvergleich:
   * Die Gate-Golden-Zahl stammt vom **train-only** gewählten Kopf über
     denselben `eval_split`-Pfad wie die Schatten-Fits. Der All-Data-Refit
     fließt in sie **gar nicht ein** (er passiert später, nur für
     `head.bin`). Meine Refit-These war damit unprüfbar falsch gestellt.
   * Prod-Pfad und Schatten-Replika benutzen **dieselbe** konsolidierte
     Spaltenfunktion — kein versteckter Spaltenunterschied.
   * Verteilungen auf fast demselben Korpus: Prod-Nightly-Seeds
     0.907–0.921, Replika-Tagesserie 0.904–0.919 → **vollständig
     überlappend, kein Pfad-Malus**. Die 0.900 vom 08-11 war ein
     gewöhnlicher Zieh-Ausreißer (Replika erreicht selbst 0.904).
     `mlp32` liegt mit 0.929–0.936 **disjunkt über beiden** — der
     Vorsprung überlebt beide Pfadvarianten.
   Offen bleibt allein das Gate selbst (Boden-Ratsche, O3) — kein
   Messpfad-Problem mehr.
3. **Kapazität (MLP-64).** → **O6, ENTSCHIEDEN 2026-08-12 noch am selben
   Abend: NICHT ERFÜLLT** (Median +0.0000, 2/5 negativ). Kapazität bringt
   auf dem nackten Kopf nichts → §4 Friedhof, zweite Beerdigung. Abschluss
   in [`serien-abschluss.json`](serien-abschluss.json).

### O3 — Läuft der Golden-Boden in eine Sperre?

*Status: Sperre aktiv seit 08-09. Seed-These 08-11 abends WIDERLEGT — bei
k=3 räumen 95 % der Ziehungen die Latte, drei Ablehnungen in Folge sind kein
Seed-Pech. Die Refit-Spur ist GESCHLOSSEN (08-12 abends, s. §3a Punkt 2):
kein Pfad-Malus, 0.900 war ein Zieh-Ausreißer. `--prod-seeds` bleibt bei 3.*

**Teilweise entschärft 2026-08-09.** Der Boden war `max()` über alle
deployten Nächte — und ein Einzelwert trägt laut Seed-Sweep bis zu 0.023
Glücksanteil. Die Sperrklinke verlangte damit, einen Glückstreffer zu
wiederholen. Seither ist der Boden **der höchste Wert, der mindestens zweimal
erreicht wurde** (zweitbester, höchstens ein Wert je Kalendertag). Wirkung:

```
  vorher   0.9213 (Maximum, 08-07)      effektive Latte 0.9113
  nachher  0.9173 (zweitbester von 3)   effektive Latte 0.9073
```

Der gestern abgelehnte Kandidat (0.9111) käme damit durch. Der Boden liegt
aber **weiterhin 0.0007 über dem Champion** (0.9166) — O3 bleibt offen, die
Änderung nimmt nur den messbaren Verzerrungsanteil heraus.

⚠️ Das ist **kein** Senken der Latte (L1). Die Latte bleibt „so gut wie schon
einmal reproduziert". Wer wirklich senken will, ändert `--golden-floor` — und
begründet es. `golden_stau()` zählt weiterhin die Nächte in Folge und warnt.

**Die Sperre ist da — und sie misst eine Ziehung, keine Drift (2026-08-11).**
Drei Nächte in Folge abgelehnt (08-09, 08-10, 08-11), alle drei am Boden, nicht
am Testsatz. In der Nacht 08-11 lief erstmals `seed_golden` mit, also der
Golden-Median **jedes** der drei Produktionsfits derselben Nacht — identische
Daten, identische Architektur, nur anderer Init-Seed:

```
  Seed 0   0.9067   ← Mitte auf dem TESTSATZ, also ausgeliefert
  Seed 1   0.9205
  Seed 2   0.9195
                     Spanne Golden 0.0138 | Spanne Testsatz 0.014
                     Boden 0.9173, Toleranz 0.010 → Latte 0.9073
```

Der ausgelieferte Kopf war auf Golden der **schlechteste der drei**; die beiden
anderen hätten die Latte geräumt. Das REJECT dieser Nacht ist damit ein
Münzwurf und keine Aussage über den Korpus. Das ist der Seed-Sweep aus §2, nur
nicht mehr als Einzelmessung, sondern als Eigenschaft jeder Nacht.

Was daraus **nicht** folgt: den Seed mit dem besten Golden ausliefern. Das wäre
Auswahl auf genau der Metrik, auf der das Gate urteilt — der Boden würde sich
selbst bestätigen und wäre als Schutz wertlos. Die Wahl per Testsatz-Median ist
bewusst so und bleibt.

Was folgt: das Gate vergleicht eine Ziehung mit ±0.014 gegen eine Schwelle mit
Toleranz 0.010 — es urteilt unterhalb seiner eigenen Auflösung. Denkbare
Antworten, keine davon umgesetzt, keine ohne eigene Messung zu haben:

* **Toleranz an das gemessene Rauschen koppeln** statt sie bei 0.010 zu
  belassen. Ehrlicher, aber lockert faktisch die Latte — braucht L1-Begründung.
* **Boden gegen die Seed-Verteilung statt gegen einen Punkt** prüfen (räumt der
  Kandidatenschwarm die Latte im Median?). Misst nicht mehr, was ausgeliefert
  wird — das ist der Einwand, an dem es hängt.
* **Mehr Seeds** (5 statt 3) verengt die Ziehung um ~√(3/5), kostet aber
  Nightly-Laufzeit linear.

Solange nichts davon gemessen ist, gilt die Sperre als **erwartetes Verhalten
unter Rauschen** und nicht als Defekt. `seed_golden` läuft ab jetzt jede Nacht
mit; nach ein paar Nächten ist die Spanne eine Verteilung statt einer Anekdote,
und dann erst lohnt die Entscheidung.

**Gemessen statt gewartet (2026-08-11 abends): der Seed ist es NICHT.**

12 Fits auf identischen Daten, nur der Init-Seed unterschiedlich, danach die
Nightly-Auswahlregel über alle Teilmengen durchgespielt (je Teilmenge den
Kopf mit dem mittleren **Testsatz**-Wert wählen, seinen **Golden**-Wert
notieren — genau die Kette, die nachts läuft):

```
  Pool: golden 0.906–0.925, Spanne 0.019, Std 0.0059

   k  Teilmengen  golden-Spanne     Std   räumt die Latte (0.9073)
   1          12         0.0190  0.0059        83 %
   3         220         0.0180  0.0047        95 %   ← heutiger Nightly
   5         792         0.0150  0.0044       100 %
   7         792         0.0150  0.0040       100 %
   9         220         0.0070  0.0024       100 %
```

⚠️ **Damit ist meine eigene Deutung vom Morgen widerlegt.** Ich hatte das
REJECT einen Münzwurf genannt. Bei k=3 räumen 95 % der Ziehungen die Latte —
drei Ablehnungen in Folge wären als reines Seed-Pech etwa 1 zu 8000. Es sitzt
also etwas **systematisch** rund 0.01 unter der Latte, und der Seed erklärt
das nicht.

**`--prod-seeds` bleibt deshalb bei 3.** Fünf Seeds kosten zwei Fits je Nacht
und kaufen eine Verbesserung am Rand — als Antwort auf O3 wären sie
irreführend, weil sie die Sperre wegräumen könnten, ohne ihre Ursache zu
berühren. Erst die Ursache, dann ggf. die Seeds.

### Die Spur: der Refit auf allen Daten

Der Produktionskopf desselben Laufs kam auf Golden **0.900** — *unterhalb des
gesamten Pools* (Minimum 0.906). Beides ist dieselbe Architektur auf
denselben Daten; der Unterschied ist, dass der Produktionskopf nach der
Auswahl **auf allen Daten neu gefittet** wird, während die Sweep-Köpfe es
nicht werden.

Wenn sich das reproduziert, ist das die Erklärung für die Blockade — und dann
ist „mehr Seeds" die Antwort auf die falsche Frage. Nächster Schritt: den
All-Data-Refit gegen den ausgewählten Kopf messen, mehrfach, nicht einmal.

⚠️ Eine Einzelmessung. Sie stammt aus einem Handlauf (`quelle="hand"`,
eigenes Archiv), nicht aus dem Nightly, und die Teilmengen oben teilen sich
Seeds — „100 %" ist eine Aussage über DIESEN Pool aus EINER Nacht, keine
Wahrscheinlichkeit.

### O4 — Sinkt der Korrekturaufwand überhaupt?

*Status: erstmals gemessen 2026-08-09, beobachten.*

`scripts/review-effort.py` misst, wie viele Sekunden je Stunde der Mensch am
Vorschlag der Maschine verschoben hat — die einzige Zahl in diesem Stack, die
nicht aus der Schleife selbst stammt. Erste Messung, nur von Menschen
reviewte Aufnahmen, nur solche außerhalb des Trainings:

```
  2026-05  n=11  Median 49.6 s/h   exakt  9 %
  2026-06  n= 9  Median 47.6 s/h   exakt 22 %
  2026-07  n= 5  Median 75.7 s/h   exakt  0 %
```

Kein sichtbarer Rückgang, bei kleinen n. Das ist **keine** Feststellung, dass
die Schleife nichts bringt — die Monatsgruppen sind zu klein und die Stichprobe
ist verzerrt (reviewt wird bevorzugt, was auffällt). Aber solange diese Zahl
nicht fällt, ist jede IoU-Verbesserung eine Behauptung über einen
Stellvertreter.

⚠️ **Nicht während einer Detect-Welle messen.** Wird die Cutlist gerade neu
geschrieben, ist die `.txt` zwischenzeitlich leer und `auto` kommt aus einer
Ersatzquelle (Anker-Fenster), die plausibel aussieht und es nicht ist.
Gemessen am 2026-08-09 mitten in einer Welle sprang derselbe Titel von 169 auf
800 s/h. Das Skript verweigert deshalb den Dienst, wenn `/api/integrity`
laufende oder wartende Detects meldet.

⚠️ **Die erste Fassung des Skripts meldete für August „100 % exakt".** Grund:
`edited=true` heißt nicht „ein Mensch war dran" — Auto-Confirm schreibt ein
synthetisches `ads_user.json`, in dem `user == auto` per Konstruktion gilt.
101 von 250 Dateien. Wer nach `auto_confirmed_at` nicht filtert, misst die
Schleife gegen sich selbst und bekommt eine perfekte Note.

### O5 — Hält, was am Golden-Satz gemessen wurde, auch anderswo?

*Status: versiegelter Satz angelegt 2026-08-09, wächst. Frühestens ab ~30
Aufnahmen öffnen.*

Der Golden-Satz ist der einzige Maßstab, und **jede** Entscheidung wird gegen
ihn getroffen. Er verliert seine Unabhängigkeit nicht durch Training, sondern
durch Wiederholung: nach genug Entscheidungen misst er, wie gut wir auf 38
Aufnahmen selektiert haben. Der versiegelte Satz ist die Gegenprobe.

**Vorwärts versiegelt** (`--sealed-frac 0.20`): 20 % aller Aufnahmen, die neu
ins Split-Ledger kommen, landen in einem dritten Eimer — weder Training noch
Auswertung, keine Entscheidung fällt gegen sie. Bestehende Aufnahmen bleiben
unberührt. Zwei Gründe:

* Ein heute aus dem Bestand geschnittener Satz enthielte Aufnahmen, gegen die
  bereits hunderte Entscheidungen gefallen sind — von Geburt an halb
  verbraucht.
* Es hätte den Korpus mitten in der laufenden O1-Serie verkleinert.

Eigener Hash-Salt (`"versiegelt:" + uuid`), damit die Versiegelung nicht mit
der Test-Zugehörigkeit korreliert. Nie versiegelt: Golden-Pins und alles
während der Ledger-Erstbefüllung.

**Die Disziplin ist der ganze Wert.** `loop-status.py` zeigt nur die *Größe*,
nie ein Ergebnis. Ihn regelmäßig auszuwerten wäre exakt die Nutzung, die ihn
wertlos macht. Öffnen: selten, und nie um zwischen Kandidaten zu wählen —
nur um zu prüfen, ob die Kurve am Golden-Satz sich anderswo wiederfindet.

**Kosten:** rund ein Fünftel der neuen Aufnahmen fehlt dem Training.
Bei ~37 Reviews im Monat sind das ~7, in einem halben Jahr ~40.

## 4. Friedhof — entschieden, nicht neu vorschlagen

**MLP-64 — zweite und letzte Beerdigung (2026-08-12, O6).** Erste Serie
(Juli, 7 Nächte) methodisch leer (fester Seed). Wiedervorlage mit
dokumentierter Begründung als Tagesserie auf dem NACKTEN Kopf: Median Δ
+0.0000, 2/5 negativ — Kapazität bringt nichts, auch nicht auf der seit O2
relevanten Architektur. Wer 64 (oder 128) erneut vorschlägt, braucht einen
Grund, der DIESE Serie entwertet, nicht nur die alte.

| Idee | Verdikt | Wann | Warum |
|---|---|---|---|
| Minute-Prior als Eingabespalte | **inert**, bleibt drin | 8 Nächte bis 2026-07 | Δ ≈ −0.001. Verdrahtung nachweislich korrekt. Nachgemessen 08-08/09 auf der *richtigen* Spalte: Δ +0.001. Werbung hängt an der Sendung, nicht an der Uhr. |
| Boundary-Head aktivieren | **negativ**, bleibt aus | 2026-07-24 | Ankerdichte zu hoch. Falle: Dumps ohne Bumper logen +0.016, faithful −0.015. |
| per-Show / per-Kanal `nn_gate`, `nn_weight`, `nn_smooth`, Snaps | **wirkungslos** | seit hsmm-Umstellung 2026-07-29 | Unter hsmm greifen nur `min/max-block-sec` und die hsmm-Priors. Alle Tunings davor (RTL-Fix etc.) sind tot. |
| MLP-64 statt MLP-32 | **Rauschen** | 2026-07-12 | 7-Nächte-Serie, kein Signal. |
| VideoToolbox für Decode | **unbrauchbar** | — | Scheitert auf interlaced H.264 aus DVB, fällt immer auf Software zurück. |
| Whisper-Engine wechseln | **entschieden** | — | whisper.cpp, 4-parallel 2.7 s gegen WhisperKit 9.5 s. |
| Deinterlace als Detect-Fix für comedy-central | **falsche Spur** | 2026-08 | NN-Signal ist dort gut (Trennung 0.646); 0.30 heilt das nicht. |
| Decoder-Sweeps, per-Show-Config, Label-Korrekturen als Hebel auf 0.915 | **ausgereizt** | 2026-08 | Einzelmessungen stehen, die Kanal-These dahinter ist widerlegt. |
| Detect-Fallback auf dem Pi | **per Design aus** | — | Pi5 ist always-on-Transport, CPU/ML läuft auf dem Mac. |

## 5. Leitplanken

Was die Schleife **nie** tut, auch nicht mit guter Begründung:

* **L1** — `--golden-floor` senken oder den Boden anders entschärfen. Die
  Sperrklinke ist genau gegen langsamen Drift gebaut; sie zu lockern, weil
  sie gerade blockt, entfernt den einzigen absoluten Maßstab.
* **L2** — Labels anfassen (`ads_user.json`, Cutlists, Golden-Satz).
  Ground truth ist Eingabe, nicht Stellschraube.
* **L3** — am Gate vorbei deployen oder `head.bin` direkt zum Gateway
  schieben.
* **L4** — aus einem Handlauf ins echte `~/.cache/tvd-train-archive`
  schreiben. Experimente laufen gegen eine Scratch-Kopie, sonst verfälschen
  sie die Historie, aus der am nächsten Tag entschieden wird.
* **L5** — den Header-Vertrag oder den Go-Inferenzpfad ändern, ohne
  Paritäts-Fixture und ohne ausdrückliches OK. Neue Spalten kommen **hinten**
  dazu, Migration ist ein Header-Bump, nie eine Umsortierung.
* **L6** — Aufnahmen oder Cache-Einträge löschen. Die Dual-Copy-Regel hat
  schon einmal zehn Aufnahmen gekostet.

## 6. Datenquellen der Schleife

| Was | Wo |
|---|---|
| Nightly-Protokoll | `~/Library/Logs/tv-train-head.log` |
| Golden-Verlauf des Kandidaten | `~/.cache/tvd-train-archive/golden-trend.jsonl` |
| Schatten-Serie je Variante | `~/.cache/tvd-train-archive/shadow-trend.jsonl` |
| per-rec-IoU (Stabilitäts-Veto) | `~/.cache/tvd-train-archive/per-rec-iou.jsonl` |
| Golden-Satz + `set_hash` | `~/.cache/tvd-train-archive/golden-eval-set.json` |
| Split-Ledger | `~/.cache/tvd-train-archive/split-ledger.json` |
| Auto vs. User-Blöcke | `GET :9984/recording/<uuid>/ads` (nie die Caches catten) |
| Mensch oder Auto-Confirm | `~/tv-labels-backup/_rec_*/ads_user.json` → `auto_confirmed_at` |

Alle liegen auf dem Mac (bzw. hinter der tv-recorder-API). Eine Schleife, die
sie liest, muss lokal laufen.

**Gesichert seit 2026-08-09:** `tv-backup-labels.sh` spiegelt
`~/.cache/tvd-train-archive` nach `~/tv-labels-backup/train-archive/` und
committet es in das private Repo. Davor lag das gesamte Gedächtnis der
Schleife — Golden-Verlauf, Golden-Satz, Test-Split, 660 eingefrorene
Archiv-Aufnahmen — ungesichert in einem Cache-Verzeichnis.
