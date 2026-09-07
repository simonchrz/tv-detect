# O21 — Vergiften widersprüchliche Frames das Training?
(Vorab-Registrierung)

**Geschrieben 2026-09-07, nachdem der Kontrollarm lief und BEVOR der
Maskenarm gerechnet wurde.**

## Woher die Frage kommt

Das Fehlerbudget (Ledger, dreizehnter Durchgang) ordnet **39 % des
IoU-Verlusts** der Label-Seite zu: Sekunden, an denen das Label harter
Evidenz widerspricht. Kein Reviewfehler — Simon reviewt vollständig. Eine
Konvention: der Review beantwortet „was muss der Spieler überspringen",
das Training liest die Antwort als „was ist Werbung".

Statt Labels umzuschreiben (L2) werden die widersprüchlichen Frames aus
dem Training genommen. `train-head.py` kennt das Muster als `frame_mask`,
mit dem Kommentar „no opinion = not training data, NOT a default-show
prediction".

## Arme

Derselbe Kopf, dieselben Merkmale, derselbe Seed, dasselbe Ziel. Der
EINZIGE Unterschied sind die Trainingszeilen.

* **Kontrolle** — alle Zeilen, wie heute.
* **Maske** — ohne die Zeilen, in denen das Label harter Evidenz
  widerspricht.

**Widerspruch** heisst nur zweierlei, beides hart:
1. Ein Wiederholungs-Anker deckt die Sekunde, das Label sagt Sendung.
   Anker-Präzision 96.8 % gegen Menschenlabel, geeicht am 2026-09-07.
2. Die Sekunde liegt in einem Endblock (endet am Aufnahmeende, >= 60 s),
   den das Label als Werbung führt.

Das Senderlogo bleibt draussen: schwächerer Zeuge, Let's Dance blendet
es aus, sixx wäscht es aus.

## Messung

F1 auf `test`, geglättet 10 s je Aufnahme, **nur auf den unbestrittenen
Zeilen** (482097 von 488002), für beide Arme identisch. Gegen die vollen
Labels zu messen hiesse, gegen genau die Labels zu messen, die unter
Verdacht stehen; der Maskenarm würde dafür bestraft, dass er Trailer
nicht mehr als Werbung lernt.

⚠️ **Was ein positives Ergebnis NICHT zeigt.** Der unbestrittene Teil
schliesst die strittigen Fälle per Konstruktion aus. „Besser" heisst
dann „besser auf dem unstrittigen Teil", nicht „besser insgesamt". Ein
Null-Ergebnis dagegen erledigt die Vergiftungs-These.

## Grösse des Eingriffs — vorher benannt

**Maskiert werden 5687 von 545913 Trainingszeilen, also 1.0 %.** Die
39 % aus dem Budget sind ein Anteil an den VERLUST-Sekunden, nicht an den
Trainingsdaten. Ein grosser Effekt wäre bei 1 % entfernter Daten
überraschend; die Schwelle ist entsprechend angesetzt, aber nicht unter
das Rauschen.

## Rauschen, gemessen vor der Schwelle

Drei Seeds des Kontrollarms: 0.9053, 0.9129, 0.9131. Median 0.9129,
**sd 0.0045**.

## Bedingung

```regel
{
  "id": "O21",
  "frage": "Vergiften widerspruechliche Frames das Training?",
  "art": "offline-kopf-ab",
  "nicht_in_serienabschluss": true,
  "metrik": "F1 auf test, geglaettet 10s, NUR unbestrittene Zeilen",
  "arme": {"kontrolle": "alle Zeilen", "versuch": "ohne widerspruechliche Zeilen"},
  "widerspruch": ["Anker deckt, Label sagt Sendung", "Endblock am Aufnahmeende"],
  "eingriff_anteil_trainingszeilen": 0.010,
  "paarung": "gleicher Seed, gleiche Architektur, gleiches Ziel",
  "seeds": 5,
  "rauschen_sd_kontrollarm": 0.0045,
  "bedingungen": {
    "median_delta_f1_mindestens": 0.002,
    "positive_seeds_mindestens": 4
  },
  "konsequenz_bei_erfuellt": "frame_mask fuer widerspruechliche Frames im Nightly vorschlagen. KEIN Deploy ohne Menschen-Entscheidung.",
  "konsequenz_bei_verfehlt": "Vergiftungs-These erledigt; die 39 % bleiben ein MESS-Problem, kein Trainingsproblem."
}
```

Beide Bedingungen müssen gelten. Die Schwelle 0.002 ist rund die Hälfte
der ungepaarten Standardabweichung; die Paarung über den Seed nimmt einen
Teil des Rauschens heraus. Die Vorzeichenbedingung trägt den Rest, sie
hängt nicht an der Effektgrösse.

**Aus O20 gelernt und hier vermieden:** dort war die Zieländerung mit
einer Gewichtsänderung vermengt, weil die Klassengewichte aus den
Häufigkeiten kommen. Hier bleibt das Ziel binär und die Gewichtsregel
dieselbe; die Häufigkeiten verschieben sich nur um die maskierten 1 %.
