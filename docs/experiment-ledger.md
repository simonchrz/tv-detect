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

*Status: **ENTSCHIEDEN 2026-08-14, REGEL NICHT ERFÜLLT.** Median −0.0044
(4/5 negativ): Vorzeichen-Bedingung erfüllt, Größen-Bedingung (≤ −0.010)
verfehlt. Nächte: −0.0206 / −0.0001 / −0.0048 / −0.0044 / +0.0048.
Konsequenz laut Registrierung: **die Whisper-Spalte bleibt** — sie schadet
nicht belegbar. Belegten NUTZEN hat sie damit aber ebenso wenig wie Kanal
(O7) und Temporal (O8); die Abwägung fällt in der Architektur-Frage.*

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

## 3y. Label-Konventionen (Simon, 2026-08-13 abends — BINDEND)

1. **Kostenpflichtige Gewinnspiel-Inserts sind SENDUNG** (winario,
   Gewinnarena — auch show-gebrandet mit 01379-Nummern).
2. **Werbung beginnt und endet meist mit einem kurzen Sendungstrailer —
   diese Rand-Trailer gehören ZUR WERBUNG.** ⚠️ Das REVIDIERT den
   End-Snap-Entscheid vom 28.07. („Trailer = Sendung"): der dortige Guard
   wurde auf der alten Lesart kalibriert und gehört neu bewertet.

Erste Anwendung: 10 Kanten in 7 Golden-Aufnahmen agent-reviewt korrigiert
(Frames gesichtet durch 4 Prüf-Agenten, Urteile mit Konfidenz; Anwendung
über den offiziellen Edit-Endpunkt, `agent_reviewed`-Marker, Backups in
scratchpad/label-review/label-backups.json). NICHT korrigiert: 3
Gewinnspiel-Spannen (Konvention 1), 1 bestätigte Sendung, 2 Übergänge mit
Kante innerhalb der Spanne.

Fünfte Anwendung (2026-08-14): **37 weitere Kanten im Rest-Korpus**
(60 Kandidaten ab 5 s, 80 Frames, 3 Agenten). Nicht angewandt: 7×
Gewinnspiel (= Sendung, Label korrekt — der Agent schrieb dort „Block zu
spät", das war seine Schlussfolgerung, nicht seine Klassifikation; die
Konvention schlägt den Kommentar), 9× bestätigte Sendung, 4 ohne Frames,
2 unsichere. Wirkung: „Sendung geschnitten" 207 → **194 s**.

Vierte Anwendung: **3 Übergänge sekundengenau aufgelöst** (dichte
1s-Abtastung, Agent benennt den ersten Frame der zweiten Seite;
konservativ nie über den Beleg hinaus). Damit ist die Label-Runde vom
13.08. RESTLOS abgeschlossen — kein aufgeschobener Fall.

Dritte Anwendung: **18 Blockanfänge vorgezogen** (24 Kandidaten/464 s
signal-hergeleitet, 51 Frames, 2 Agenten: 10× Werbung, 8× Trailer/
gemischt-Block, 3× Gewinnspiel korrekt belassen, 1× Kante innerhalb,
2 ohne Frames). Damit ist die Rand-Trailer-Konvention in BEIDE
Richtungen über den Korpus angewendet: ~47 Kanten an einem Abend,
agent-reviewt, Null manuelle Review-Minuten.

Zweite Anwendung (gleicher Abend): **19 Blockenden verlängert** — die
Gegenrichtung der 07-28-Kürzungen. Kandidaten signal-hergeleitet (21
Spannen/329 s, fast alle ProSieben), 40 Frames von 2 Agenten gesichtet:
18× Trailer, 1× Werbung, 2× Sendung (unangetastet). Backups kumulativ in
label-backups.json.

⚠️ R1-Vermerk: die Golden-Ground-Truth dieser 7 Aufnahmen hat sich damit
geändert — Golden-Werte ab dem Nightly 14.08. messen gegen korrigierte
Labels. Die gepaarten O1-Arme sehen beide dieselben Labels, die Paarung
bleibt fair.

## 3s. Gewinnspiel-Inserts: kein Label-Rauschen, sondern eine
Konventions-Lücke im Modell (2026-08-14)

Runde 2 der Kanten-Review (28 kurze Streitspannen) hat den Grund geliefert,
warum die Label-Runden das VOLUMEN halbierten, aber die Zusammensetzung
nicht verschoben: **6 der 28 Spannen waren Gewinnspiel-Inserts** (winario,
GewinnArena, 01379-Nummern). Das Modell liest sie mit p>0.7 als Werbung —
nach Konvention §3y sind sie **Sendung**, das Label ist also korrekt.

⚠️ **Damit ist die Kategorie „Label-Verdacht" (77 %) systematisch
überzeichnet.** Ein erheblicher Teil davon ist gar kein Label-Fehler,
sondern ein **systematischer, LERNBARER Modellfehler**: eine Klasse, die
das Modell nie beigebracht bekommen hat. Sie ist visuell hoch distinktiv
(statische Tafel, Telefonnummer, Preis-Overlay) — ein besserer Kandidat
für eine echte Modell-Frage als alles, was diese Woche gemessen wurde.

Angewandt Runde 2: 9 von 28 Kanten (6× Gewinnspiel und 7× bestätigte
Sendung korrekt NICHT angewandt).

## 3t. Gewinnspiel gemessen — und der Detektor lohnt NICHT (2026-08-14)

Nachmessung meiner eigenen Empfehlung von gestern („bessere Modell-Frage
als alles diese Woche Gemessene"). Sie hält nicht.

**Größe der Klasse.** Über den Golden-Satz gibt es 12 Spannen „Auto sagt
Werbung, Label sagt Sendung", zusammen **183 s = 0.34 % der Laufzeit**
(14.9 h). Frame-Review dieser 12:

| Kategorie | Spannen | Sekunden | Bewertung |
|---|---|---|---|
| GEWINNSPIEL | 3 | 68 s | Label richtig, Modell irrt |
| WERBUNG | 3 | 67 s | **Label falsch, Modell hatte recht** |
| TRAILER | 3 | 13 s | **Label falsch** (Rand-Trailer = Werbung, §3y) |
| SENDUNG | 2 | 11 s | Label richtig, Modell irrt |
| ohne Frames | 1 | 25 s | — |

Gewinnspiel ist also **37 % der Streitmasse, nicht die Mehrheit** — meine
Formulierung „ein erheblicher Teil" war zu stark. Die eigentliche Hälfte
sind **falsche Labels, bei denen das Modell recht hatte** (80 s).

**Trennbarkeit.** Lineare Sonde auf dem Backbone (1280 dim), 7 bekannte
Gewinnspiel-Spannen aus 6 Aufnahmen, leave-one-RECORDING-out: **AUC-Median
1.000**. Das klingt entschieden, ist es aber nicht — balanciert gemessen,
effektive Stichprobe sind 7 Spannen, nicht 116 Sekunden. Der ehrliche Test
gegen die GANZE Zeitachse der ausgelassenen Aufnahme:

```
kabel-eins  55 % erkannt, 10 s/Std Fehlalarm
kabel-eins 100 %,         49 s/Std
prosieben  100 %,         64 s/Std
prosieben  100 %,         38 s/Std
prosieben  100 %,         14 s/Std
vox          0 % (!),      2 s/Std
```

VOX fällt komplett aus — hält man die einzige VOX-Aufnahme zurück, kennt
das Modell die winario-Aufmachung nicht mehr. Die Sonde lernt **Marken,
keine Klasse**.

⚠️ **Verdikt: nicht bauen.** Die Klasse ist 68 s auf 15 h ≈ **4.5 s/Std** —
kleiner als die Fehlalarmrate des Detektors selbst (2–64 s/Std). Ein
Aufwand, der mehr Fehler erzeugt als er behebt. Bliebe der Nutzen, wenn es
zehnmal mehr Beispiele gäbe und die Marken-Verallgemeinerung trüge; das
ist derzeit nicht belegt und die Obergrenze bleibt 0.34 % der Laufzeit.

**Angewandt:** 5 der 6 Label-Fehler (75 s; einer mit niedriger
Agent-Sicherheit ausgelassen). Korpusweit „Sendung weggeschnitten"
194 s → 119 s.

## 3ab. O10 (OCR-Programmhinweis am Blockrand) — NICHT ERFÜLLT an H1,
aber die Diagnose ist wertvoll (2026-08-14)

Registrierung: [`o10-ocr-hinweis-preregistration.md`](o10-ocr-hinweis-preregistration.md),
committet vor der ersten Zahl (1346f75).

Beste Zelle (Fenster 60 s, Nachlauf 5 s): Median-IoU **+0.0191** — weit
über der geforderten Schwelle von +0.005 — aber **3 Aufnahmen besser gegen
4 schlechter**. Die zweite H1-Bedingung ist verfehlt, also NICHT ERFÜLLT.
Der Bestätigungssatz blieb unangetastet.

**Die Diagnose ist der eigentliche Ertrag.** Von 40 Blockenden tragen nur 9
einen Programmhinweis, und diese 9 teilen sich perfekt:

* **3 Gewinner:** hsmm-Ende 32/42/69 s zu früh, Hinweis direkt dahinter.
  Genau der konstruierte Fall — die Regel repariert ihn.
* **4 Verlierer:** hsmm-Ende sitzt exakt auf dem Label (0/−7/−8/0 s) und
  trotzdem folgt ein Programmhinweis.

Nach Konvention §3y wären diese vier LABEL zu kurz. ⚠️ Das darf O10 nicht
retten — gemessen wurde gegen `ads_user.json`, und ein nachträglicher
Labelwechsel zugunsten der eigenen Regel ist genau die Manipulation, gegen
die die Registrierung existiert. Wer weitermachen will: die vier Kanten per
Frames klären, korrigieren, und die Frage auf **unbeteiligten** Aufnahmen
neu registrieren.

**Dritter gescheiterter Decoder-Weg für dieselbe Klasse** (nach §3x
Randverlängerung 16:12 und §3z Ad-Bias 6:11). Anders als die beiden
scheitert dieser nicht an der Idee, sondern an der Referenz.

## 3ai. Das Logo ist doch ein gutes Signal — meine Absage war zu
breit (2026-08-15, Einwand des Users)

⚠️ **Korrektur.** In §3af steht „das Logo rettet keine dieser Kanten, blend
ist damit erledigt". Gemessen war das ausschließlich auf den **76
Fehlerkanten** — eine nach Konstruktion verzerrte Auswahl: dort versagen
per Definition beide Signale. Über den ganzen Korpus sieht es anders aus.

**Trennschärfe über 105 Aufnahmen:**

| | im Werbeblock | in der Sendung | klar getrennt | Überlappung |
|---|---|---|---|---|
| Logo | 0.104 | 0.915 | 85 % | 1 % |
| NN | 0.935 | 0.039 | 100 % | 0 % |

Das Logo trennt also sauber — der User hatte recht. Und für die KANTE sind
beide gleich gut: Logo-Flanke Median 3.2 s, NN-Flanke 3.3 s. Entscheidend
ist aber, dass sie **komplementär** sind: das Logo liegt bei 37 % der
Kanten näher, das NN bei 43 %, gleich bei 19 %.

**Trotzdem bringt `hsmm-blend` nichts** (105 Aufnahmen, Kantenfehler):

```
        hsmm    Median 4.2s  p75 13.0s  p90 34.6s  <=2s 37%  Summe 4073s
  hsmm-blend    Median 4.0s  p75 15.0s  p90 36.1s  <=2s 39%  Summe 4955s
```

Marginal besserer Median, schlechterer Schwanz, 20 % mehr Gesamtfehler. Der
Grund steckt in der Mechanik: blend **mittelt** die Emissionen
(`ad = 1-((1-w)*logo + w*(1-nn))`). Zwei Signale, die jeweils in
VERSCHIEDENEN Fällen recht haben, verlieren beim Mitteln beide. Gebraucht
wird eine **Auswahl**, keine Mischung.

**Wieviel in der Auswahl steckt** (Fenster ±30 s um die DECODER-Kante, also
einsetzbar — nicht um die wahre Kante, das wäre Nachbetrachtung):

| | Median | p75 | ≤2 s | Summe |
|---|---|---|---|---|
| Decoder (Ist) | 4.4 s | 14.8 s | 36 % | 4448 s |
| Heuristik „größere Flanke" | 3.5 s | 13.5 s | 38 % | 4281 s |
| Orakel (perfekte Auswahl) | 2.5 s | 9.9 s | 45 % | 3773 s |

⚠️ Der erste Anlauf legte das Fenster um die WAHRE Kante und ergab Orakel
1.5 s / 58 % — das ist Unsinn, weil ein Verfahren im Betrieb nicht weiß, wo
die Wahrheit liegt. Die Tabelle oben ist die ehrliche Fassung.

**Einordnung:** eine naive Heuristik holt ~0.9 s Median, die Obergrenze
perfekter Auswahl liegt bei ~2 s Median und 45 % statt 36 % auf 2 s. Real,
aber kein Durchbruch — und es adressiert die harten Fälle NICHT (§3ah: 29 %
der schlechten Kanten haben gar keinen Anker, und bei Trailern/Split-Screen
irren beide Signale gemeinsam). Eine Auswahl zwischen zwei Signalen hilft
nicht, wo beide falsch liegen.

## 3ah. Warum die Kante schwer ist — gemessen (2026-08-15)

336 Kanten, ohne die zeitversetzte Aufnahme.

**Was NICHT hilft:** Schwarzbilder gibt es praktisch nicht (2–7 %, egal wie
gut die Kante sitzt) — die Sender schneiden hart ohne Schwarzblende. Stille
trennt auch nicht (36 % / 25 % / 31 %).

**Was trennt: der Szenenschnitt.**

| Kante | n | Schnitt ±2 s | nn-Sprung | Logo-Sprung |
|---|---|---|---|---|
| genau (≤2 s) | 121 | **83 %** | 0.98 | 0.98 |
| mittel (2–20 s) | 153 | 61 % | 0.95 | 0.97 |
| daneben (>20 s) | 62 | **48 %** | 0.85 | 0.74 |

**Und die eigentliche Schwierigkeit — es sind zu VIELE Schnitte.** Über
84 h liegen 26134 Szenenschnitte, also einer alle **12 s**. Im ±15-s-Fenster
um die wahre Kante stehen im Median **3** Kandidaten, bei knapp einem
Drittel der Kanten mehr als 5.

Damit zerfällt „ungenau" in zwei ganz verschiedene Probleme:

1. **Auswahl** — es gibt einen Anker, aber mehrere. Eine Kante auf 2 s zu
   treffen heißt, unter 3–4 gleich aussehenden Schnitten den richtigen zu
   wählen. Ein Snap-Verfahren hilft hier nur, wenn es weiß, WELCHER.
2. **Kein Anker** — bei **29 %** der schlecht sitzenden Kanten gibt es im
   ±15-s-Fenster überhaupt keinen Szenenschnitt (gegen 5 % bei den genauen).
   Der Übergang ist weich: Überblendung, Split-Screen-Auflösung, oder die
   Sendung läuft unter einer stehenden Grafik einfach weiter. Da ist im Bild
   nichts, worauf man springen könnte.

Das erklärt rückblickend, warum die deterministischen Snaps historisch so
uneinheitlich gemessen haben (§3x, `hsmm_refine.go`): sie adressieren
Problem 1 und machen Problem 2 schlimmer, und beide sind im selben Korpus
vermischt.

## 3ag. Ein Drittel der Kantenfehler-Masse war ein Zeitachsen-Versatz
(2026-08-15)

Bei der Arbeit an Punkt 4 (Let's Dance, 623 s Fehler aus einer Aufnahme)
fiel die Form auf: acht Blöcke, alle mit **richtiger Länge** (Label
492–521 s, Auto 518–611 s), alle **~90–120 s zu spät**. Ein Modell irrt
nicht so gleichförmig.

**Ursache:** die Labels entstehen auf der **VOD**-Zeitachse (die App zeigt
das HLS), die Merkmale auf der **Quell-.ts**. Bei
`dvr-rtl-1780078500` ist das VOD **16090 s** lang, die Quelle **16202 s** —
**112 s Versatz**. Bestätigt über die OCR-Marker: der nächste Werbe-Marker
lag konstant +111 s nach dem Label-Start, also bei Quelle 5398 für einen
Label-Start bei VOD 5287 — nach Umrechnung **exakt** dieselbe Stelle.

**Was das kostet:**

| | Kanten | Median | p90 | Fehlermasse > 20 s |
|---|---|---|---|---|
| mit der Aufnahme | 352 | 5.0 s | 44.2 s | **4511 s** |
| ohne sie | 336 | 4.4 s | 35.6 s | **3089 s** |

**Ein Drittel des Schwanzes war Messartefakt aus einer einzigen Aufnahme.**
Und schlimmer als die Messung: der Eintrag liegt im **TRAIN**-Split mit
Archiv-`.npz` — die verschobenen Labels haben den Kopf mittrainiert.

### Entwarnung, aber ein Detektor

Über 249 Aufnahmen mit Labels: **eine** über der Schwelle. Drei weitere bei
exakt +6.0 s = eine HLS-Segmentlänge, also Endsegment-Rundung — die
verschiebt keine Grenze und wird nicht quarantäniert. Die Schwelle steht
deshalb bei 15 s (mehr als zwei Segmente), nicht bei 5.

⚠️ Die Kennzahl sagt NICHT, wo die Differenz sitzt. Fehlt am Ende ein
Segment, verschiebt das nichts; fehlt am Anfang etwas, verschiebt es alles.
Deshalb ist die Schwelle grob und der Bericht zeigt auch, was sie
durchlässt.

**Gebaut:** `scripts/zeitachsen-check.py` schreibt die Liste (braucht das
Gateway — der Snapshot führt `index.m3u8` nur als 0-Byte-Marker),
`train-head.py` liest sie und lässt die Aufnahmen aus dem Korpus, laut
protokolliert. Läuft im Nightly VOR dem Training.

**Lehre:** eine Fehlerform, die zu gleichförmig für ein Modell ist, ist
keine Modellfrage. Ich habe erst Emission, Decoder und per-Show-Konfiguration
durchgesehen, bevor ich auf die Blocklängen geschaut habe — die Antwort
stand von Anfang an in der Tabelle.

## 3al. Der grobe Durchgang: das Modell startet Werbeblöcke
SYSTEMATISCH zu früh (2026-08-16, erste zwei Aufnahmen)

Als Reparatur des Auswahlfehlers aus §3ak: die ganze Aufnahme im 45-s-Takt
abtasten (80–95 Bilder je Stunde) und die Blöcke daraus ableiten, **ohne**
die Annahme, dass das Modell ungefähr richtig liegt.

```
kabel-eins-1786546385 (64 min)
   Modell 1552-2101   grob 1620-2115   Start +68s   Ende +14s
   Modell 2763-3310   grob 2835-3330   Start +72s   Ende +20s

prosieben-1786035922 (72 min)
   Modell 2487-2963   grob 2520-2970   Start +33s   Ende  +7s
   Modell 3490-3988   grob 3510-4005   Start +20s   Ende +17s
   Modell 4147-4310   grob 4185-4275   Start +38s   Ende -35s
```

**Fünf von fünf Blockstarts liegen zu früh, um 20 bis 72 Sekunden.** Die
Enden sind deutlich näher dran. Die Richtung ist über beide Aufnahmen und
alle Blöcke gleich.

⚠️ **Warum das die kantenbezogene Bauart nicht finden konnte:** sie sucht
±40 s um die Modellkante. Ein Startfehler von 68 s liegt außerhalb und fällt
als „kein Wechsel im Fenster" durch — genau so sind RTL, VOX und zweimal
ProSieben aus dem Messsatz verschwunden, während die gut sitzenden
Aufnahmen durchkamen. Der Auswahleffekt aus §3ak ist damit nicht mehr eine
Vermutung aus der Ausgangslage, sondern belegt.

### Und es ist die Gewinnspiel-Klasse, größer als gedacht

Das Modell startet zu früh, weil es die **Mitmachtafel vor dem Block** als
Werbebeginn liest — nach Konvention ist sie Sendung (§3y, vom User am
2026-08-16 an drei Bildern bestätigt). Genau dieses Muster zeigte sich schon
bei jeder kabel-eins-Aufnahme im Agenten-Review.

⚠️ **Damit ist mein Urteil vom 2026-08-14 zu korrigieren.** In §3t hatte ich
die Gewinnspiel-Klasse mit „68 s über den Golden-Satz, 4.5 s/Std — kleiner
als die Fehlalarmrate eines Detektors, nicht bauen" abgelegt. Diese Zahl
stammt aus dem Golden-Satz und misst nur, was dort an Streitmasse zwischen
Auto und Label liegt. Der grobe Durchgang zeigt **eine Minute je Blockstart**
— und beim Ad-Skip wird das von der Sendung abgeschnitten, was irreversibel
ist. Die Klasse ist damit wieder offen und deutlich größer als gemessen.

⚠️⚠️ **KORREKTUR am 2026-08-17: die Beträge sind um ~22 s überzeichnet.**

Der grobe Durchgang setzt den Blockstart auf den ersten Werbe-Rasterpunkt.
Die wahre Grenze liegt irgendwo im 45-s-Intervall davor — das Verfahren
meldet also **im Mittel 22 s zu spät**, per Konstruktion und unabhängig
davon, ob die Referenz stimmt. Ich hatte oben „20 bis 72 s zu früh"
geschrieben, ohne diesen Versatz abzuziehen.

Belegt wurde er, indem derselbe Durchgang gegen die MENSCHLICHEN Labels des
Golden-Satzes lief (3 Aufnahmen, 4 Blöcke): Starts Median **+29 s**, Enden
**+8 s**, nur 1 von 4 Grenzen über der 45-s-Auflösung. Die +29 s bei
Grenzen, die ein Mensch gesetzt hat, sind praktisch der
Quantisierungsversatz — sie sagen nichts über die Labels.

**Nach Abzug** bleiben von den fünf Modell-Blöcken:
kabel-eins **+45 s / +49 s** (weiterhin groß), prosieben **−2 / +11 / +16 s**
(im Rauschen). Die Richtung stimmt also, aber der Befund trägt nur noch für
kabel-eins, nicht als allgemeine Aussage über den Korpus.

**Was der grobe Durchgang kann und was nicht:** er findet zuverlässig
Fehler ÜBER 45 s (bei kabel-eins-1778511363 im Golden-Satz einen von +54 s)
und ist damit das Werkzeug gegen den Auswahlfehler aus §3ak. Für alles
darunter ist er blind, und wer ihn dort auswertet, misst sein eigenes
Raster. Wer genaue Beträge will, braucht einen zweiten, feinen Durchgang an
den grob gefundenen Stellen — dieselbe Zwei-Stufen-Logik wie beim
Nachfassen.

**Vorbehalt bleibt:** zwei Aufnahmen für den Modellvergleich, drei für den
Label-Vergleich. Bevor daraus eine registrierte Frage wird, gehört das über
deutlich mehr Aufnahmen — und mit der feinen zweiten Stufe.

## 3ak. O14 NICHT ERFÜLLT — und der Messsatz bildet den Korpus nicht ab
(2026-08-16)

76 Kanten aus 20 Aufnahmen, prospektiv gesammelt, alle Labels von
Review-Agenten.

```
  [1] Median 0.0s -> 0.5s        VERFEHLT
  [2] <=2s 67% -> 63%            VERFEHLT
  [3] Gesamtfehler 482s -> 496s  VERFEHLT
  gewaehlt: nn 23, logo 53
```

Alle drei Bedingungen verfehlt. Vorhersage war „nicht erfüllt, knapp" —
richtig im Ausgang, falsch im „knapp".

⚠️ **Der Befund über das Ergebnis hinaus:** die Ausgangslage ist Median
**0.0 s** und **67 %** der Kanten auf 2 Sekunden. Korpusweit sind es 4.4 s
und 36 % (§3af). Der Schattensatz ist also viel besser als der Korpus — weil
eine Aufnahme nur ins Ledger kommt, wenn JEDE ihrer Kanten aus den Bildern
bestimmbar war. Genau daran scheiterten die schlecht sitzenden Aufnahmen
(RTL: 168 Sendungsbilder gegen 22 Werbebilder selbst im ±72-s-Fenster; VOX;
zweimal ProSieben).

**Die Ablehnung, die falsche Labels verhindert, filtert zugleich die
interessanten Fälle heraus.** O14 hat die Flankenauswahl damit an genau den
Kanten geprüft, an denen nichts zu gewinnen ist. Das Urteil steht (so war es
registriert), aber es belegt nur: auf schon korrekten Kanten schadet die
Regel.

**Konsequenz für künftige Messungen an Kanten:** ein Satz, der um die
Modellkante herum gebaut wird, kann per Konstruktion keine grob falschen
Kanten enthalten. Wer die will, muss die Kante erst grob über die ganze
Aufnahme suchen und dann verfeinern.

## 3aj. Agenten-Review: WELCHE Frage man stellt, entscheidet alles
(2026-08-16, zwei menschliche Stichproben)

Zwei Bauarten, dieselben Agenten, gegensätzliche Ergebnisse — geprüft vom
User an insgesamt sieben Einzelbildern.

**Bauart A — Spannen klassifizieren (14./15.08.): 4 von 4 richtig.**
Kandidaten aus einer SIGNAL-Uneinigkeit ableiten (Auto-Block gegen Label),
Frames aus der Spanne zeigen, fragen „was ist hier zu sehen", und nur
eindeutige Kategorien anwenden. 92 verschobene Kanten in 65 Aufnahmen;
Stichprobe A/B/C/D (sat-1-gold +190 s, ProSieben +104 s, VOX, kabel eins)
vom User bestätigt.

**Bauart B — Kante bestimmen (16.08.): 2 von 3 falsch.** Frames um die
Blockkante zeigen und fragen „bei welcher Sekunde ist der Übergang".

  * Blockende auf „Sendung läuft wieder" gesetzt, obwohl über das ganze
    Fenster Trailer lief — belegt mit einem Vorspann, den es dort nicht gibt.
  * Kante unverändert bestätigt, obwohl der Übergang einen Abtastpunkt
    früher lag.

⚠️⚠️ **KORREKTUR meiner eigenen Auswertung.** Der dritte Fall stand hier
zunächst als schwerster: 4 von 4 kabel-eins-Blockstarts wurden mit
„GewinnArena, 01379-Nummer, 4×25.000 €" abgelehnt, und ich las die Antwort
des Users („das ist alles Werbungsübergang") als Widerspruch — also als
konsistente Halluzination, ausgelöst davon, dass mein Auftrag die
Markennamen selbst aufzählte. **Das war falsch.** Der User beschrieb, was zu
sehen ist, nicht wo der Block beginnt. Auf Nachfrage mit drei Bildern hat er
die Kante auf **1594** gelegt — also hinter die Tafel, genau wie der Agent
gesagt hatte. Der Agent lag richtig, die Konvention `mitmachtafel = Sendung`
ist bestätigt, und die „Halluzination" gab es nie.

Die Lehre daraus ist eine über MICH, nicht über den Agenten: eine
Beschreibung („was ist im Bild") ist keine Entscheidung („wo gehört die
Grenze hin"). Ich habe die eine als die andere gelesen und daraufhin einen
Befund gemeldet, ein Werkzeug für kaputt erklärt und Labels
zurückgenommen. Wer eine Stichprobe auswertet, muss vorher sagen, welche
Frage sie beantwortet.

⚠️ **Der eine Konstruktionsfehler, der bleibt:** die Frage erzwingt eine
Antwort. „Bei welcher Sekunde" hat immer eine; „ich sehe die Grenze nicht"
muss der Agent von sich aus sagen, und er tut es nicht zuverlässig. Genau
das erledigt jetzt `kante_aus_folge` maschinell.

**Die Lehre ist übertragbar:** ein Agent ist gut darin zu sagen, WAS auf
einem Bild ist, und schlecht darin, eine Grenze zwischen Bildern zu
bestimmen. Das erste ist eine Klassifikation mit sichtbarem Beleg, das
zweite ein Urteil über Abwesenheit („ab hier nicht mehr"). Wer das zweite
braucht, muss es aus dem ersten ABLEITEN — pro Bild klassifizieren lassen
und die Grenze selbst aus der Folge bestimmen, statt sie zu erfragen.

Die vier Labels aus Bauart B sind zurückgenommen; die 92 aus Bauart A
bleiben.

## 3af. Die Kanten direkt gemessen — und die Fehlermasse zerlegt
(2026-08-15)

⚠️ **Methodischer Kern: Block-IoU war die falsche Kennzahl.** Sie misst
überwiegend Block-MASSE. Das Ziel sind Kanten. Direkt gemessen über 104
Aufnahmen mit frischen Signalen, 344 Kanten:

| | Median | p75 | p90 | ≤2 s | ≤5 s | ≤10 s |
|---|---|---|---|---|---|---|
| Starts | 4.3 s | 16.0 s | 39.8 s | 35 % | 54 % | 66 % |
| Enden | 5.0 s | 21.2 s | 54.2 s | 35 % | 51 % | 62 % |

**Blöcke FINDEN ist gelöst** — von 173 Label-Blöcken wird genau einer ganz
verpasst (15 Auto-Blöcke haben keinen Partner). Das gesamte Problem ist die
Kantenlage. IoU 0.937 verdeckt, dass nur ein Drittel der Kanten auf 2 s
sitzt.

### Die Fehlermasse: 76 Kanten > 20 s tragen 4454 s

| Gruppe | Kanten | Sekunden | Frame-Befund |
|---|---|---|---|
| Kopf sicher WERBUNG (nn>0.85), Wahrheit Sendung | 9 | 973 s | 4× Let's-Dance-Live-Inhalt, 2× Trailer |
| Kopf sicher SENDUNG (nn<0.15), Label sagt Werbung | 15 | 896 s | **10× Label zu großzügig**, 4× Split-Screen-Werbung, 1× Trailer |
| Mittelband 0.15–0.85 | 52 | 2585 s | (nicht einzeln geprüft) |

Bei den 15 „sicher Sendung"-Fällen liegt **ausnahmslos** das Label weiter
außen — kein einziges Mal umgekehrt.

### Drei Befunde, die vorher nicht sichtbar waren

**1. Der Juli-Fix für Let's Dance ist tot.** `dvr-rtl-1780078500` trägt
allein 623 s der ersten Gruppe. Die per-Show-Konfiguration setzt
`nn_gate 0` / `nn_weight 1` (Fix vom 07-28 gegen das Logo-Ausblenden).
⚠️ **Im Code nachgewiesen wirkungslos:** Produktion fährt plain `hsmm`, und
dort ist `emit := nnConf` (`cmd/tv-detect/decoder.go:119`) — `NNWeight`
liest **nur** `hsmm-blend`/`hsmm-full`, `nn_gate` gar nichts. Seit dem
Decoder-Wechsel am 07-29 ist der Fix inert, ohne dass es auffiel. Bestätigt
Memory `hsmm_ignoriert_per_show_tuning` am Code statt aus der Erinnerung.

**2. Die Logo-Spur ist kein Hebel — gemessen, nicht vermutet.** Plain
`hsmm` ignoriert das Logo vollständig; `hsmm-blend` würde es einmischen.
In den Fehlerbereichen sagt es aber dasselbe wie das NN: bei „sicher
Werbung" Median 0.196 (Logo sagt auch Werbung — Let's Dance blendet sein
Logo während der Show aus), bei „sicher Sendung" 0.907 (sagt auch Sendung —
Trailer und Split-Screen tragen das Senderlogo). Im Mittelband widerspricht
es zu 48 %, also Münzwurf. **blend rettet keine dieser Kanten.** Die Idee
ist damit billig erledigt.

**3. Split-Screen-Werbung ist eine eigene Fehlerklasse.** Bei Let's Dance
laufen Spots im geteilten Bild, während die Show in einer Ecke weiterläuft.
NN liest Sendung (nn 0.03–0.18), Logo liest Sendung — und es IST Werbung,
mit der Kennzeichnung „Werbung" plus Countdown im Bild. Beide vorhandenen
Signale versagen gemeinsam, per Konstruktion.

### Wieviel davon steht im Bild geschrieben

OCR über alle 76 Lücken (2 Frames je Lücke, also eine UNTERGRENZE):

```
  Programmhinweis (Wochentag+Uhrzeit)    13 Luecken   796s  (18 %)
  Werbe-Kennzeichnung ("Werbung")         5 Luecken   370s  ( 8 %)
  mindestens eines von beiden            18 Luecken  1167s  (26 %)
  ohne jeden lesbaren Marker             58 Luecken  3287s
```

**26 % der Kantenfehler-Masse liegt dort, wo der Bildschirm buchstäblich
hinschreibt, was läuft** — und zwar in Information, die weder NN noch Logo
trägt (§3aa: 224×224 verwirft sie). Beispiele: die Split-Screen-Spots
zeigen „WERBUNG" bei nn=0.03; eine 179-s-Lücke zeigt „Montag 20:15" bei
nn=0.88.

⚠️ Kein Freifahrtschein: 74 % tragen keinen lesbaren Marker, und OCR ist
kein Ersatz für das Modell. Aber es ist der einzige gemessene Kandidat, der
Information HINZUFÜGT statt vorhandene umzurechnen — und er ist jetzt in
Sekunden auf der richtigen Kennzahl beziffert statt in
IoU-Nachkommastellen.

⚠️ **Nicht vergessen: 10 der 15 geprüften „sicher Sendung"-Lücken waren
LABEL-Fehler** (zu großzügig). Ein Teil der gemessenen 4454 s ist also gar
kein Modellfehler. Die Korrektur steht aus und gehört gemacht, BEVOR die
nächste Frage auf diesen Zahlen registriert wird.

## 3ae. O12 — der eine Schuss: NICHT ERFÜLLT um 0.0083 (2026-08-15)

Registrierung [`o12-ocr-bestaetigung-preregistration.md`](o12-ocr-bestaetigung-preregistration.md),
committet vor der Messung (abfb2c8). 22 nie gemessene Golden-Aufnahmen,
6 von der Regel angefasst.

```
  [1] Median-Delta angefasst : +0.0307   ERFUELLT  (>= +0.010)
  [2] besser 5 / schlechter 1            ERFUELLT
  [3] groesster Einzelverlust: -0.0583   VERFEHLT  (> -0.05)
```

Bedingung 1 dreifach übererfüllt, Bedingung 2 klar — und trotzdem NICHT
ERFÜLLT, weil eine einzelne Aufnahme 0.0083 über der Verlustgrenze liegt.
„Sendung weggeschnitten" 270 s → 302 s (+12 %, Grenze 20 % gehalten).

⚠️ **Die Schwelle wird nicht nachverhandelt** und die eine Verlierer-Aufnahme
nicht per Frames „geprüft". Eine Grenze, die nach dem Blick auf die Zahl
neu gesetzt wird, ist keine Grenze.

**Bilanz der Reihe:** vier Registrierungen, vier Fehlschläge, jeder an
einer anderen Bedingung — Sache (O9), Referenz (O10), Kennzahl (O11),
Verlustgrenze (O12). Die Messlage ist konsistent positiv (angefasste
Aufnahmen +0.020 bei 7:3, dann +0.031 bei 5:1), aber kein Durchgang hat
seine eigene vorher gesetzte Latte geräumt. Der Bestätigungssatz ist für
diese Regel verbraucht; ein fünfter Satz wäre das Muster, das §3ad benannt
hat.

**Offener, sauberer Weg:** die Regel im Nightly als SCHATTEN mitlaufen
lassen (gemessen, nicht angewandt) und prospektiv auf Aufnahmen sammeln,
die es beim Registrieren nicht gab. Das ist kein neuer Anlauf auf denselben
Daten, sondern der Verzicht darauf.

## 3ad. O11 — NICHT ERFÜLLT, aber die Kennzahl war meine Wahl
(2026-08-14)

Registrierung [`o11-ocr-fluechtig-preregistration.md`](o11-ocr-fluechtig-preregistration.md),
committet vor der Messung (4fc5971). 60 unbeteiligte Aufnahmen,
stratifiziert über 11 Kanäle; 56 auswertbar, 101 Blockenden, 13 mit
Hinweis, **1 dauerhaft** (die Flüchtigkeits-Bedingung greift genau einmal).

```
Regel MIT Fluechtigkeit   Median +0.0028   besser 7   schlechter 3   Verlust -0.049
```

Zwei von drei Bedingungen erfüllt (7:3, kein Verlust über 0.10), die erste
verfehlt: **+0.0028 statt +0.005.** NICHT ERFÜLLT. Bestätigungssatz
unangetastet.

⚠️ **Der Median über alle 56 ist für einen spärlichen Eingriff die falsche
Kennzahl** — die Regel fasst 11 Aufnahmen an, 45 bleiben unverändert, der
Median sitzt in den Unberührten. Auf den 11 angefassten: **Median +0.0201,
Mittel +0.0415, 7 besser / 3 schlechter, Spanne −0.049 bis +0.190.** Die
Schwelle +0.005 stammt aus O9, wo ein globaler Regler jede Aufnahme
anfasste; sie unbesehen zu übertragen war mein Fehler — und er ändert das
Urteil nicht, dafür existiert die Registrierung.

⚠️ **Das Muster ist das eigentliche Warnsignal:** drei Registrierungen,
drei Fehlschläge, drei verschiedene Gründe (Sache / Referenz / Kennzahl).
Jeder war echt und protokolliert, aber wer oft genug neu registriert,
gewinnt irgendwann zufällig. Konsequenz: **keine vierte Abstimmungsrunde.**
Es bleibt ein Kandidat mit plausibler Wirkung und genau eine unverbrauchte
Instanz — der Bestätigungssatz. Ein Schuss, vorab festgelegte Bedingung,
Ergebnis gilt.

## 3ac. Der Fehlermodus des OCR-Musters: DAUER-Einblendungen
(2026-08-14, gefunden durch O10s Verlierer)

Frame-Review der vier O10-Verlierer spaltet sie sauber:

* **3× TRAILER** (prosieben, „Dienstag 21:25", „Mittwoch 20:15",
  „Sonntag 20:15" + ProSieben-/Joyn-Ident) → die **Labels waren zu kurz**,
  die Regel hatte recht. Korrigiert (+13/+9/+13 s; alle drei liegen im
  Abstimmungssatz, NICHT im Golden-Satz — kein Epochenwechsel).
* **2× SENDUNG** (rtlzwei-1779016500, „Von Hecke zu Hecke") → die Sendung
  läuft, und oben rechts klebt ein **Dauer-Programmhinweis** „NEUE FOLGE
  HEUTE 20:15 / KAMPF DER REALITY ALLSTARS" durch die ganze Sendung.

⚠️ **Das ist ein echter Falschtreffer-Modus, und meine Negativkontrolle hat
ihn nicht gefunden** (0 von 120 zufälligen Sendungs-Punkten aus 93
Aufnahmen, §3aa). Die Kontrolle war nicht falsch, sondern
unterdimensioniert für ein seltenes, aber SYSTEMATISCHES Muster: trifft es
eine Aufnahme, dann gleich über die volle Laufzeit — genau deshalb trug
rtlzwei allein den größten Einzelverlust (−0.144).

**Die Unterscheidung ist mechanisch einfach und noch ungetestet:** ein
Trailer-Hinweis ist **flüchtig** (10–30 s), eine Dauer-Einblendung steht
Minuten bis Stunden. Eine Regel, die den Treffer nur zählt, wenn dasselbe
Muster 120 s früher NICHT da war, würde rtlzwei verwerfen und die drei
prosieben-Fälle behalten.

⚠️ Das gehört in eine NEUE Registrierung auf **unbeteiligten** Aufnahmen.
Es nachträglich in O10 einzubauen und dann „bestanden" zu melden, wäre die
Manipulation, gegen die die Vorab-Registrierung existiert — zumal die
Referenz inzwischen von mir selbst korrigiert wurde.

## 3aa. Der Backbone kann die Einblendung nicht LESEN — und genau
darin steckt das verallgemeinerbare Merkmal (2026-08-14)

Nach zwei gescheiterten Decoder-Wegen (§3x, §3z) die Frage andersherum
gestellt: **warum** gibt der Kopf Trailern nur 0.586? Die Antwort ist eine
Überanpassungs-Lücke, und sie ist klassenspezifisch:

| | train | test | Lücke |
|---|---|---|---|
| Sendung | 0.031 | 0.008 | +0.023 |
| Werbung (Kern) | 0.930 | 0.922 | +0.007 |
| **Trailer** | **0.684** | **0.433** | **+0.251** |

Zehnmal so groß wie bei allem anderen. Der Kopf **merkt sich** gesehene
Trailer und verallgemeinert nicht — er lernt „dieses Filmmaterial ist
Werbung", nicht das Merkmal.

**Warum er es nicht lernen kann:** der Backbone bekommt 224×224
(`internal/signals/nn.go:112`), skaliert aus 1920×1080. Eine
Programmhinweis-Leiste ist darin ~8 px hoch — physisch unlesbar. Das
unterscheidende Merkmal eines Trailers ist aber genau dieser Text.

### Gegenprobe mit macOS-Vision-OCR (`tools/ocr/ocr.swift`)

Muster: Wochentag/„Heute"/„MO-FR"/„Jeden Tag" **und** eine Uhrzeit.

| Menge | Treffer |
|---|---|
| TRAILER-Spannen (agent-belegt) | **10 von 11** |
| SENDUNG-Spannen direkt nach Blockende | **0 von 32** |
| Zufällige Sendungs-Punkte mitten in 93 Aufnahmen | **0 von 120** |

Bei den zufälligen Sendungs-Frames findet die OCR in 34 % *irgendeinen*
Text — sie funktioniert also, findet nur nie das Muster. Beispiele aus den
Trailern: „Samstag 20:15", „Heute 20:40 Uhr", „FREITAG 00:00",
„GHOSTS MO-FR 19:15", „JEDEN TAG 14:00", „Heute 7:45 / Neu!".

**Kosten:** 32 ms/Frame einfädig. Kanten-lokal (±90 s um jede Blockgrenze,
~720 Frames) sind das ~23 s je Aufnahme = **+17 %** auf die Detect-Kosten
(Backbone allein braucht ~135 s). Volle Abdeckung bei 1 fps wären 134 s —
das wäre zu teuer, kanten-lokal ist es nicht.

⚠️ **Warum das NICHT unter §3w fällt** („abgeleitete Spalten fügen nichts
hinzu"): OCR ist keine Umrechnung vorhandener Merkmale. Es ist Information,
die der 224×224-Eingang **physisch verwirft**. Das ist der erste Hebel
dieser Session, der nicht strukturell tot ist.

**Nächster Schritt, mit eigener Registrierung:** die O9-Registrierung hat
für den Fall des Scheiterns bereits die *kanten-lokale Emission* als
nächsten Kandidaten benannt — und die hat jetzt einen Inhalt. Der billige
Weg zuerst: OCR an den Blockgrenzen der 32 Aufnahmen mit frischen
Signal-Dumps, Regel „verlängere den Block über einen Programmhinweis",
gemessen mit derselben Abstimmung/Bestätigung-Trennung wie O9. Erst wenn
das trägt, lohnt die teure Variante (eigene Spalte, Neu-Extraktion des
Korpus).

## 3z. O9 (Ad-Bias) — NICHT ERFÜLLT an H1 (2026-08-14)

Registrierung: [`o9-adbias-preregistration.md`](o9-adbias-preregistration.md),
geschrieben und committet VOR der ersten Zahl (cd82e6e).

Der beste von sieben Bias-Werten bringt auf dem Abstimmungssatz (22
Aufnahmen, alle NICHT im Golden-Satz) **+0.0004** Median-IoU — gefordert
waren +0.005 — bei 6 besseren gegen 11 schlechtere Aufnahmen und −0.184
größtem Einzelverlust. Der Bestätigungssatz blieb unangetastet.

⚠️ Meine registrierte Vorhersage („H1 besteht") war falsch. Zusammen mit
der Randverlängerung (§3x, 16:12) ergibt sich dieselbe Lehre zweimal:
**ein globaler Regler trifft jede Kante, und die richtigen sind in der
Überzahl.** Die Trailer-Klasse ist damit gut vermessen (356 s, 24 s/Std,
Kopf trennt sie sauber) und decoder-seitig auf zwei Wegen nicht erreichbar.
Offene Kandidaten wären der Dauer-Prior oder eine kanten-lokale Emission —
je mit eigener Registrierung.

## 3x. Kopf oder Decoder? DECODER — belegt. Die naheliegende
Randverlängerung trägt aber nicht (2026-08-14)

17 Aufnahmen mit Trailer-Befund frisch durchgerechnet (Produktionsbefehl
via `redetect` + `.want`-Marker, nicht rekonstruiert). Der Signal-Cache
taugte nicht: bei allen 14 vorhandenen war `nn_confs: null`.

**Die Frage ist entschieden.** Über 19 Aufnahmen und 21 Trailer-Spannen:

| | Median | 10.–90. Perzentil |
|---|---|---|
| Sendung | **0.009** | 0.006 – 0.036 |
| Trailer | **0.586** | 0.219 – 0.845 |
| Werbung (Kern) | **0.921** | 0.851 – 0.946 |

**21 von 21 Trailern liegen über dem 90. Perzentil der Sendung.** Bei
Schwelle 0.15 werden alle erfasst, ohne dass eine einzige Aufnahme über
ihrem Sendungsmittel läge. Der Kopf trennt Trailer sauber von Sendung —
der Decoder verwirft die Information. Auf der Einzelaufnahme
`dvr-prosieben-1778926191` liegt der 75-s-Trailer bei 0.406 zwischen
Blockkern 0.927 und Sendung 0.008, und fällt bei 1153 s schlagartig auf
0.012: die Kante ist im Signal frame-genau da.

### Die Regel dazu — drei Fassungen, alle nicht tragfähig

„Block ab dem hsmm-Ende weiterlaufen lassen, solange die Konfidenz über
einer Schwelle bleibt." Auf der Einzelaufnahme spektakulär: 75 s Fehler →
2–3 s. Über 31 Aufnahmen und 92 Kanten:

1. **Produktions-Glättung (symmetrisch 10 s):** schleppt die hohe
   Blockkonfidenz über die Kante — 478 s Abdrift an KORREKTEN Kanten gegen
   363 s Gewinn. Netto negativ.
2. **Ungeglättet + Hysterese:** Trailer haben interne Dips, der Lauf bricht
   zu früh ab — nur 97 s Gewinn.
3. **Gerichtete Glättung** (am Ende nur das Vorwärtsfenster, am Start nur
   das Rückwärtsfenster): überbrückt die Dips, ohne dass der Block ins
   Fenster fällt. Beste Zelle (Schwelle 0.20, Fenster 10 s, **nur Enden**):
   Gesamt-Kantenfehler 1469 s → 1394 s, **−75 s (5 %)**, 16 Kanten besser
   gegen 12 schlechter.

⚠️ **Nicht deployen.** 16:12 ist fast ein Münzwurf, und 5 % Gesamtfehler
rechtfertigen keinen neuen Sonderweg im Blockformer. Die Startseite ist in
JEDER Zelle negativ — das Phänomen sitzt am Blockende, wie erwartet.

⚠️ **Messfalle, die ich zweimal fast bezahlt hätte:** die ersten beiden
Auswertungen zählten „Gewinn an schiefen Kanten" gegen „Abdrift an
richtigen Kanten" — und lasen die Regel dadurch als Nullsummenspiel. Eine
Verschiebung an einer schon guten Kante ist aber kein Verlust, wenn sie
zum Label hin geht. Richtig ist der GESAMTFEHLER über alle Kanten. Die
Zahlen oben sind die korrigierte Rechnung.

**Was daraus folgt:** die Information liegt sauber im Kopf-Ausgang, also
gehört sie in den Decoder — aber nicht als handgeschnitzter Lauf, sondern
in die Emission selbst, die ein 0.2–0.85-Band derzeit wie Sendung behandelt.
`--hsmm-ad-bias` (bereits in `hsmm.go`, mit Test) ist genau dieser
Stellhebel und ist bisher nur exploratorisch vermessen. Das ist der nächste
saubere Schritt — mit Registrierung vorab, weil es ein Produktionsparameter
ist.

## 3w. Die andere Fehlerrichtung: 65 % der verpassten Werbung sind
TRAILER — und der Korpus ist NICHT schuld (2026-08-14)

Bisher immer nur „Sendung geschnitten" angeschaut. Die Gegenrichtung ist
4,7-mal größer. Frame-Review aller 24 Spannen „Label sagt Werbung, Modell
sagt Sendung" über den Golden-Satz (545 s):

| Kategorie | Spannen | Sekunden | |
|---|---|---|---|
| TRAILER | 11 | **356 s** | Label richtig, Modell verpasst |
| WERBUNG | 4 | 54 s | Label richtig, Modell verpasst |
| SENDUNG | 7 | 86 s | Label zu großzügig |
| unklar | 2 | 49 s | |

**65 % der gesamten verpassten Werbung sind Trailer** — 24 s/Std, fünfmal
so groß wie die Gewinnspiel-Klasse aus §3t.

⚠️ **Erste These widerlegt.** Ich hatte vermutet: der Korpus lehrt noch die
alte Konvention (§3y-Wechsel am 08-13), die 36 am 07-28 auf die NN-Kante
gekürzten Blöcke vergiften das Training. Die 36 sind per `reviewed_at`
nicht auffindbar (die Juli-Aktion hat den Zeitstempel nicht angefasst, die
Scratchpad-Backups von damals sind weg), also stattdessen **40 zufällige
Blockenden im Korpus** (ohne Golden) mit Frames geprüft — was läuft in den
14 s nach einem Blockende?

```
SENDUNG 32 | WERBUNG 4 | TRAILER 2 | ohne Frames 2
```

**Nur 5 % haben noch einen Trailer dahinter.** Die Korpus-Labels sind an
den Enden sauber; die 356 s sind eine echte Modellschwäche, kein
Konventionskonflikt. Die Juli-Kürzung ist entweder längst überschrieben
oder war nie so verbreitet wie die Notiz nahelegt.

### Trennbarkeit — und warum eine neue Spalte NICHT hilft

Lineare Sonde auf dem Backbone, 22 agent-belegte Trailer-Spannen aus 20
Aufnahmen und 8 Kanälen, leave-one-RECORDING-out gegen Sendung derselben
Aufnahme:

```
AUC Median 0.835   min 0.453   max 1.000   unter 0.7: 5/20
```

Deutlich schwächer als Gewinnspiel (1.000) — erwartbar, Trailer SIND
Programmmaterial. Ein Viertel der Aufnahmen liegt bei Zufall.

⚠️ **Der strukturelle Punkt, der für BEIDE Klassen gilt:** eine lineare
Sonde auf demselben Backbone misst, was der Kopf ohnehin lernen KANN. Der
Produktionskopf ist ein MLP auf genau diesen Merkmalen. Eine daraus
abgeleitete „Trailer-Spalte" oder „Gewinnspiel-Spalte" fügt **keine
Information hinzu** — sie rechnet um, was schon da ist. Das erklärt
rückblickend auch O2/O6/O7/O8 (alle NICHT ERFÜLLT): Spalten aus vorhandenen
Merkmalen sind ausgereizt.

**Offen und als nächstes zu messen:** liegt es am KOPF (Merkmale tragen die
Trailer nicht) oder am DECODER (der Kopf sieht sie, hsmm glättet eine 14-30 s
Kante am Blockrand weg)? Das entscheidet über völlig verschiedene Fixes.
Der Signal-Cache taugt nicht dafür — bei allen 14 Trailer-Aufnahmen mit
Cache ist `nn_confs: null` (ohne NN evaluiert). Es braucht einen frischen
detect-Lauf mit NN auf einer Aufnahme mit belegtem verpasstem Trailer.

## 3v. `set_hash` sichert die Zusammensetzung — NICHT die Labels
(2026-08-14, Lücke geschlossen)

⚠️ Beim Anwenden von 3t ist die eigentliche Lücke aufgefallen. Der
Golden-Satz trägt seit v2 einen `set_hash`, und Boden, Gate, Tagesbericht
und loop-status vergleichen nur Zeilen mit gleichem Hash **und** gleichem
Decoder. Der Hash deckt aber ausschließlich die **Mitgliederliste** ab.

**Wer die LABELS eines Mitglieds korrigiert, verschiebt den Maßstab, ohne
dass eine einzige Kennzahl es anzeigt.** Genau das ist am 2026-08-13
passiert: 87 Kanten-Korrekturen, Golden 0.906 → 0.937. Als Arbeit richtig,
als MESSUNG aber ein Schnitt — und die Latte ist mitgestiegen, als hätte
das Modell zugelegt. Der Sprung steht bis heute unmarkiert im Trend.

**Fix (deployed):** `golden_label_hash()` hasht die Blöcke aller
Mitglieder; der Wert steht ab sofort als `label_hash` in **jeder**
Trendzeile (Produktion, Shadow, Tagesserie) und ist Teil des Filters in
`golden_bestwert`, `golden_stau`, `tagesbericht.boden_und_champion` und
`loop-status`. Eine Label-Änderung schneidet die Reihe damit sichtbar, statt
sie still zu verlängern. Der Tagesbericht meldet dann „keine Latte — Labels
geändert, Reihe beginnt neu"; das Gate fällt für eine Nacht auf, wie bei
einem Satz- oder Decoder-Wechsel auch (dokumentiertes Verhalten).

Zwei Tests in `test_golden_boden.py`: Epochen-Schnitt und `"*"`-Altpfad.
Epoche vor der Korrektur `cef19e8b50ad`, danach `e6ddc1d84786`.

⚠️ **Falle beim Einbau (fast bezahlt):** die erste Fassung hängte
`label_hash` an eine Variable, die nur unter `args.shadow_eval` existiert —
der Produktions-Schreiber hätte einen NameError geworfen. Dieselbe Bauart
hat schon einmal zehn Fits ohne Zeilen produziert (`_gmeta`). Geteilte
Zwischenvariablen über Zweiggrenzen hinweg sind in diesem Skript der
wiederkehrende Fehler.

## 3u. Golden-Satz vergrößern: NICHT an der Review-Kapazität blockiert
(gemessen 2026-08-14)

Der Plan war, die 63 unreviewten Test-Eimer-Aufnahmen per Agent-Review zu
Golden-Kandidaten zu machen. **Gemessen: davon haben nur 3 überhaupt einen
Auto-Vorschlag UND Features im Snapshot.** Der Engpass ist also nicht die
Review-Zeit (die lösen die Agenten), sondern **Datenverfügbarkeit**: der
Trainings-Snapshot umfasst 285 von ~790 Ledger-uuids; die übrigen
Test-Aufnahmen liegen gar nicht als auswertbares Material vor (Pi-.ts
dedupliziert, keine Features, kein Auto-Vorschlag).

⚠️ Meine Empfehlung von gestern („Agenten lösen den Golden-Engpass") war
damit zu schnell — sie hat den Review-Engpass gelöst und den echten nicht
gesehen. Ein größerer Golden-Satz braucht zuerst mehr auswertbares
Material, nicht mehr Urteile.

**Geklärt 2026-08-14:** der Snapshot listet nur `_rec_*`-Verzeichnisse,
die auf der Pi NOCH EXISTIEREN und eine Cutlist haben. Der Ledger sammelt
uuids dauerhaft (803), die Pi-Verzeichnisse räumt die Serien-Retention
(295 übrig). Kein Defekt.

⚠️ **Korrektur einer eigenen Behauptung:** ich hatte gesagt, jede Woche
Warten koste Golden-Kandidaten an die Retention. Das ist überzogen —
Labels liegen als `<uuid>.npz` im Trainings-Archiv, Features in
`tvd-features`, beides auf dem Mac und unabhängig von der Pi. **16 der 38
Golden-Aufnahmen haben auf der Pi kein Verzeichnis mehr und werden
trotzdem 38/38 gewertet, genau darüber.** Verloren geht nur, was NIE in
den Korpus kam.

⚠️ Das ist zugleich eine stille Abhängigkeit: für diese 16 ist das Archiv
die einzige Kopie des MASSSTABS. Der Archiv-Rsync greift (16/16 in
`~/tv-labels-backup/train-archive/`) — ohne ihn wäre ein Archivverlust
nicht nur Trainingsmaterial, sondern das Messinstrument selbst.

**Stand der Kandidaten (2026-08-14):** 7 pinbar (Mensch-Label, Material,
Features). 16 weitere wären agent-reviewbar — aber das sind
auto-bestätigte, also systematisch die LEICHTEN Fälle; sie in den Maßstab
zu pinnen hübe den Golden-Median aus Gründen, die nichts mit Modellgüte
zu tun haben. Empfehlung: nur menschlich gelabelte pinnen, und erst wenn
genug für EINEN sauberen `set_hash`-Wechsel zusammenkommen (7 → Rauschen
×0.92, das lohnt den R1-Bruch noch nicht).

## 3t. Der Umschalter hätte die Messung mitgerissen (gefunden 2026-08-14)

Zwei Folgen des Architekturwechsels, beide erst beim Nachsehen sichtbar,
beide vor der ersten Nacht behoben:

1. **Der volle Spaltensatz wäre aus der Reihe gefallen.** Er wurde nie als
   Schatten-Sonde geführt, sondern nur als `baseline`-Zeile — also als
   das, was `--head-arch` gerade IST. Ab `mlp32` hätte niemand mehr
   gemessen, wogegen der Wechsel sich beweisen muss. Jetzt eigene Sonde
   (`_augment_voll`, Arch-Name unverändert in der jsonl).
2. **Die übrigen Sonden hätten still ihre Bedeutung gewechselt.**
   `wants_churn = wants_whispermask` hängt am laufenden `--head-arch`;
   unter `mlp32` wäre die Unruhe-Spalte aus `mlp32-cwt`, `mlp32-cwt-mp`
   und `mlp32-ct-mp` gefallen — **gleicher Name, andere Spaltenzahl**,
   also genau die stille Neudefinition, gegen die R1 steht, und mitten in
   den O1-Armen. Die drei Sonden sind jetzt fest verdrahtet (`churn=True`,
   was der bisherigen Bedeutung entspricht: im Nightly war `--head-arch`
   immer mp-wm).

⚠️ **Regel daraus:** eine Messgröße darf nicht davon abhängen, was gerade
Produktion ist. Der Produktionspfad (`_prod_zusatz`, Seed-Sweep,
Tagesserien-Prod-Arm) folgt `wants_*` weiterhin — dort ist es richtig.

Breiten gegengeprüft: voll 1288 Spalten, nackt 1280.

## 3v. End-Snap-Guard: DORMANT, nicht fehlkalibriert (geprüft 2026-08-14)

Nach der Trailer-Konvention (§3y) stand der Verdacht, der
`--bumper-end-nn-guard` vom 28.07. kürze jetzt aktiv in die falsche
Richtung. **Geprüft: nein.** `snapToBumperGuarded` wird ausschließlich in
`Form()` gerufen, und Produktion fährt seit 2026-07-29 `hsmm` — das
wendet per Design KEINE deterministischen Snaps an. Der Guard ist damit
dormant, genau wie die per-Show-Configs.

Zwei Konsequenzen:
* **Kein Handlungsbedarf am Decoder.** Der Guard schadet heute nichts.
* **Der echte Rest der Juli-Entscheidung sind die LABELS**: 36 Blöcke
  wurden damals auf die alte Lesart gekürzt (−1840 s). Das ist
  Label-Schaden, kein Decoder-Schaden — und genau das räumen die
  Agent-Review-Runden seit 13.08. ab (19 Enden verlängert, weitere in
  Arbeit).

⚠️ Wer den `form`-Pfad je reaktiviert (Fallback ohne NN-Konfidenzen!),
muss den Guard vorher neu bewerten: er ist auf „Trailer = Sendung"
kalibriert und damit gegen die geltende Konvention.

## 3w. Architekturwechsel auf den nackten Kopf — SCHARF seit 2026-08-14

**Simons Entscheid, ausdrücklich GEGEN meine Empfehlung.** Die Beleglage
und der Widerspruch, damit später niemand rekonstruieren muss:

```
Nacht        bare     prod    Δ(bare−prod)
20260810    0.9297   0.9057      +0.0240
20260811    0.9235   0.9067      +0.0168
20260812    0.9362   0.9101      +0.0261   ← O2 gemessen (ERFÜLLT)
20260813    0.9233   0.9164      +0.0069   ← Label-Runde beginnt
20260814    0.9332   0.9372      −0.0040   ← nach der Label-Runde
```

**Der bare-Arm hat sich nicht verändert (~0.93 durchgehend); die
PRODUKTION ist auf 0.9372 gestiegen.** Deutung: die Zusatzspalten haben
nicht „geschadet", sie haben die Kanten-INKONSISTENZ der Labels gelernt —
mehr Kapazität für Widersprüche heißt mehr angepasstes Rauschen. Mit
sauberer Ground Truth wird dieselbe Kapazität zum Vorteil. **O2s Messung
war zum Zeitpunkt korrekt, ihre Konsequenz ist damit überholt** (die
Serie selbst bleibt gültig und im Abschluss verbucht — sie wurde nicht
nachträglich umgedeutet).

Meine Empfehlung war: nicht wechseln, der Wechsel verschlechtert die
Produktion um ~0.004. Simon hat anders entschieden; umgesetzt per
`TVH_HEAD_ARCH_OVERRIDE` im Nightly-Wrapper.

⚠️ **Das GATE entscheidet weiterhin** (R5). Der bare Kopf ist heute Nacht
nur KANDIDAT: Head-to-Head gegen den Champion plus Golden-Boden. Kommt er
nicht durch, bleibt der bisherige Kopf deployt — der Wechsel der
Trainings-Architektur ist kein Deploy.

**Zurückdrehen** = `TVH_HEAD_ARCH_OVERRIDE=""` im Wrapper. Eine Zeile.

**Woran das zu beurteilen ist:** golden-trend ab 15.08. gegen 0.9372, und
die Fehlermoden-Zerlegung — nicht an einer Einzelnacht (Std 0.006).

## 3z. Ziel und Außenmetriken (festgeschrieben 2026-08-13)

**Das Ziel ist nicht eine IoU-Zahl, sondern: eine Aufnahme ansehen, ohne an
Werbung zu denken und ohne vorher zu prüfen.** Daran gemessen ist die
ehrlichste Zahl die **Exakt-Rate** aus O4 (~12 % bei IoU 0.936, Median
~50 s/h Korrektur): der Median ist fast perfekt, aber fast jede Aufnahme hat
noch irgendeinen kleinen Fehler — und fürs Erlebnis zählt „null Fehler oder
nicht", nicht „wie nah dran". Dazu sind die Kosten **asymmetrisch**:
Sendung wegschneiden wiegt vielfach schwerer als Werbereste stehen lassen;
die IoU behandelt beides gleich.

Konsequenz: neue Serien nennen, wo sinnvoll, die Exakt-Rate als Zweitmaß
neben dem Golden-Median. Eine Änderung des GATES auf ein anderes Maß wäre
dagegen ein eigener, registrierter Schritt (R1/L1) — nicht nebenbei.

## 3b. Roadmap über die Spalten-Fragen hinaus (2026-08-13, mit Simon)

1. **~~Implizite Labels aus dem Abspielverhalten~~ — BEERDIGT 2026-08-13
   (Simons Entscheid, §4).** Das Nutzungsprofil trägt es nicht:
   hauptsächlich Live-TV, Aufnahmen selten; bei Live-Werbung wird nicht
   gespult, sondern PiP/weggeschaltet — reichlich Signale, aber
   einseitig (nur Anfänge), sekunden-unscharf, systematisch verspätet.
   Für ein Kanten-Problem mit ±20 s kein Maßstab. Gestorben VOR dem
   Bauen, an einer Frage an Simon — genau dafür war Zählen-vor-Bauen da.

   **Ersetzt durch: Kanten aus dem MATERIAL (Simons Richtung:
   „autonomer/automatisierter Ansatz").** Die einzige Kante, die
   beidseitig und ohne Reaktionszeit vorliegt, ist die im Signal selbst:
   Logo-Ein/Aus, Tonpegel, Schwarzbilder. Beleglage (2026-08-13):
   * Kantenfehler-Verteilung: 57 verschobene Kanten, Median 6,1 s —
     **40 % ≤5 s** (Snap-Reichweite), 54 % ≤10 s; Schwanz bis 75 s =
     Trailer/Ident-Semantik (End-Snap-Guard-Klasse, Trailer = Sendung
     per Nutzer-Entscheid).
   * ⚠️ Vorarbeit existiert UND ist tot zugleich: Snaps auf
     Schwarzbilder gab es im `form`-Decoder — **unter hsmm sind sie
     seit 2026-07-29 wirkungslos** (Memory `hsmm_ignoriert_per_show_
     tuning`). Der Mechanismus muss als NACH-hsmm-Verfeinerung neu
     entstehen, nicht als Wiederbelebung der form-Snaps.
   * **Anker-Messung 2026-08-13 abends: NEGATIV** (`kanten_anker.py`).
     Nur 42 % der wahren Kanten haben einen Logo-Anker in ±5 s; 35 %
     der Auto-Kanten sitzen SCHON auf einem Anker; ein naiver Snap
     macht 13 Kanten schlechter und nur 7 besser (Median 6,1 → 7,5 s).
     Das Problem ist Anker-AUSWAHL (Trailer-Zonen: Logo an, Mensch
     zählt schon Werbung), nicht Anker-Suche — **exakt der Grund, aus
     dem der Boundary-Head im Juli beerdigt wurde („Anker-Dichte zu
     hoch")**. Ich habe die Friedhofs-Verbindung erst NACH der Messung
     gezogen statt vor dem Vorschlag; die Messung hat den Fehlbau
     verhindert, der Friedhof hätte ihn billiger verhindert.
     → Material-Snap v1 ist damit TOT. Vorderste Kanten-Frage bleibt
     Nr. 5 (asymmetrische Kosten) — die braucht keine Anker. Daneben
     Label-Audit der Trailer-Zonen: ein Teil der 57 „Fehler" kann
     Label-Inkonsistenz sein (Trailer am Blockanfang als Werbung
     gezählt, am Blockende als Sendung — vgl. End-Snap-Guard).
2. **Selektive Autonomie.** Kalibrierte Sicherheit routet: sichere
   Aufnahmen ohne Review-Angebot, unsichere in die Queue. Metrik:
   Abdeckung bei null Fehlern. Produktsprung aus vorhandenen Teilen.
3. **Episoden-Gedächtnis (zurückgestellt 2026-08-13 mittags).** Meine
   Begründung („der Schwanz zeigt auf Struktur") hat die Fehlermoden-
   Zerlegung noch am selben Tag widerlegt: über den ganzen Golden-Satz
   **30× Grenze, 0× verpasst, 0× Phantom** — es gibt keine Struktur-
   Fehler, die ein Episoden-Prior heilen könnte. Wieder vorlegen nur mit
   Belegen aus VERPASST-Fällen (z. B. außerhalb des Golden-Satzes).
4. **Spot-Datenbank als Erstklasse-Signal — geprüft 2026-08-13, trägt
   NICHT für die Kanten.** DB existiert (20 402 Fenster / 6 847
   Familien), aber wiederkehrende Familien liegen nur an 3 % der
   falschen Kanten in ±5 s (Median 39 s): die Fehlkanten sitzen in
   Trailer-/Ident-Zonen ohne wiederholte Spots. Für Blockfindung
   überflüssig (7 Phantom / 2 Verpasst in 223). Bleibt als Idee für
   ANDERES (z. B. Intro-Skip), nicht für die Kanten-Front.
5. **Asymmetrische Decoder-Kosten — GEPARKT 2026-08-14: die Frage hat
   kein Ziel mehr.** Nach 87 korrigierten Kanten neu gemessen:
   „Sendung geschnitten" im Golden-Satz 202 s → **95 s**, aber die
   Zusammensetzung ist praktisch unverändert — **77 % weiter
   Label-Verdacht (p>0.7), 18 % ambivalent, 5 s (5 %) echter
   Modellfehler.** Fünf Sekunden über den ganzen Golden-Satz sind kein
   Optimierungsziel; eine Bias-Serie würde weiterhin gegen Label-Rauschen
   messen. Der Regler bleibt gebaut und getestet, die Serie bleibt
   ungeschrieben.

   ⚠️ Das ist zugleich das Maß für die Label-Arbeit: die Runden haben das
   VOLUMEN halbiert, nicht die Zusammensetzung verschoben. Es liegt also
   noch eine Runde bereit — aber die braucht eine ANDERE Kandidaten-
   Herleitung: bisher über zusammenhängende p>0.55-Läufe ab 5 s, die
   verbliebenen Fälle sind kürzer und werden über den Streitspannen-
   Mittelwert sichtbar.

   *Ursprünglich:* **Regler gebaut (`--hsmm-ad-bias`),
   Serie aber ZURÜCKGESTELLT nach dem Kanten-Label-Audit (13.08. abends):
   85 % der „Sendung geschnitten"-Sekunden sind LABEL-VERDACHT (Signal
   p>0.7 in der Streitspanne), echte Modellfehler: 1 s im ganzen
   Golden-Satz. Eine Bias-Serie hätte gegen Label-Rauschen optimiert.
   Vorher: Verdachtsliste reviewen (/tmp/kanten-label-verdacht.md —
   nur die Spannen, Minuten Aufwand), dann neu messen.** Ursprünglich:** `scripts/fehlermoden.py` über den Golden-
   Satz: der gesamte Restfehler sind Grenzverschiebungen, davon 301 s
   weggeschnittene Sendung (Einzelfälle −20,6 s / +18 s) gegen 522 s
   harmlos stehende Werbung. „Nie Sendung schneiden" als HSMM-Kosten,
   gemessen an Sendung-geschnitten-Sekunden und Exakt-Rate — nicht an
   der IoU, die beide Richtungen gleich bestraft.

**Neue Fähigkeiten mit derselben Technik** (Backlog, je eigener Entwurf):
Intro/Outro-Skip über Episoden-Matching · Kapitelmarken für Magazine ·
inhaltsbasierter Anfang/Ende-Beschnitt · Schadens-Routing auf die
Mediathek-Kopie.

## 3a. Warteschlange

Eine Frage ist **eingereiht**, sobald ihre Registrierung existiert und
`serie_ab` in der Zukunft liegt; das Audit führt sie dann als „Serie hat
noch nicht begonnen". Es braucht kein eigenes Queue-Format — die
Registrierung IST der Eintrag, und damit gilt R4 (Regel vor den Daten)
automatisch auch für alles, was noch wartet.

Was ansteht, in dieser Reihenfolge:

1. **Welche Spalte genau? → O7/O8 ENTSCHIEDEN 2026-08-13 früh, beide
   NICHT ERFÜLLT.** Kanal: Median −0.0005 (3/5 negativ). Temporal:
   Median +0.0005 (2/5 negativ). Keine Einzelspalte schadet messbar —
   aber keine hat einen belegten Beitrag.

   ⚠️ **Der eigentliche Befund steht zwischen den Serien:** O2 misst für
   die SUMME der Spalten −0.025, die Einzelsonden messen ~0, ~0, ~−0.005
   (Whisper, Serie läuft), inert (Minute-Prior). Die Teile addieren sich
   nicht zum Ganzen — der Schaden entsteht im ZUSAMMENSPIEL (plus Maske,
   die an O1 hängt), nicht in einer toxischen Einzelspalte. Für die
   Architektur-Entscheidung heißt das: es gibt keinen Kandidaten für
   „diese eine Spalte entfernen und der Rest bleibt" — die belegte
   Alternative ist der nackte Kopf, und laut beiden Registrierungen
   bleibt eine Null-Spalte nur, wenn Einfachheit gegen belegten Nutzen
   abgewogen FÜR sie ausgeht. Belegten Nutzen hat derzeit keine. Je eine
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

**Verhaltensbasierte Labels (Wiedergabe-Signale) — beerdigt 2026-08-13,
vor dem Bauen.** Entwurf und Format waren fertig verhandelt (App meldet
roh, Recorder klassifiziert; /tmp/wiedergabe-signale-BACKEND.md v2).
Gestorben am Nutzungsprofil, nicht am Entwurf: hauptsächlich Live-TV, dort
wird bei Werbung nicht gespult, sondern PiP/weggeschaltet — Signale ohne
Enden, mit Reaktionszeit-Verzerrung. Wiedervorlage nur, falls sich das
NUTZUNGSPROFIL ändert (z. B. Aufnahmen werden Hauptnutzung) — nicht mit
einem besseren Format, das Format war nie das Problem.

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
