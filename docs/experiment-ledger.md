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

| gemessen | Verfahren | Golden Std | Golden Spanne |
|---|---|---|---|
| 2026-08-09 | 5 Seeds, Produktions-Architektur, identische Daten | **0.008** | **0.023** (0.901–0.924) |

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

*Status: offen, noch keine Sonde.*

Im Handlauf 2026-08-09 hatte die schlankste Variante überhaupt — `MLP-32`
ohne jede Zusatzspalte — den höchsten Golden-Median der Tabelle (0.938 gegen
0.920 der Produktion). Das ist eine Einzelmessung und widerspricht der
Migrations-Historie (Kanal und temporal wurden über Serien positiv
gemessen). Die Frage verdient dieselbe Isolationsbehandlung wie O1, aber
**nach** O1 — zwei Serien gleichzeitig auf demselben Rauschboden sind nicht
auseinanderzuhalten.

### O3 — Läuft der Golden-Boden in eine Sperre?

*Status: beobachten.*

Der Boden steht auf 0.921 (08-07), der Champion auf 0.917 (08-08). Der Boden
liegt damit **über** dem, was der amtierende Kopf selbst erreicht. Solange
das so ist, kommt nur noch durch, wer den Champion schlägt. `golden_stau()`
zählt die Nächte in Folge und warnt — die Warnung ernst nehmen, aber
**nicht** mit einer Erhöhung von `--golden-floor` beantworten (Leitplanke L1).

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

## 4. Friedhof — entschieden, nicht neu vorschlagen

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
