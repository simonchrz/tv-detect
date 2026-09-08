# O22 — Hilft die Lautheits-SCHWANKUNG, wo die Lautheit selbst nichts bringt?
(Vorab-Registrierung)

**Geschrieben 2026-09-08, nachdem der Kontrollarm lief und BEVOR der
Versuchsarm gerechnet wurde.**

## Woher die Frage kommt

Die Spalten-Wichtigkeit am deployten Kopf zeigt die Audio-Spalte als
praktisch unbenutzt: Permutationsverlust **0.0021** gegen 0.2954 beim
Logo und 0.4481 für die gesamte Backbone-Gruppe.

Die Spalte ist nicht kaputt — sie trägt echte, schwankende Werte. Kaputt
ist ihre **Prämisse**. Ihr Docstring begründet sie mit „ads ~6-10 dB
hotter than show content". Gemessen an 293576 Sekunden aus 98 Aufnahmen:
**1.23 dB.** Die EU-Lautheitsregulierung hat den alten Trick erledigt.

Dieselben Rohdaten, andere Statistik, sagen mehr. Werbung ist stark
komprimiert und schwankt kaum; Sendung hat Dialog, Musik und Stille.
AUC **innerhalb** jeder Aufnahme, Median über 98:

| Statistik | AUC | nützlich (>0.6) in |
|---|---|---|
| Lautheit (wie heute) | 0.592 | 43 % |
| Schwankung über 10 s | 0.695 | 80 % |
| **Schwankung über 30 s** | **0.726** | **81 %** |

⚠️ Je Aufnahme, nicht gepoolt. Gepoolt misst man zum Teil Unterschiede
ZWISCHEN Sendern statt innerhalb — dieselbe Häufungsfalle, die am
2026-09-07 aus einem Faktor 1.8 einen Faktor 3.3 gemacht hat.

## Arme

Derselbe Kopf, dieselben Zeilen, derselbe Seed. Einziger Unterschied ist
eine zusätzliche Eingabespalte.

* **Kontrolle** — 1282 Spalten, wie heute.
* **Versuch** — 1283 Spalten: zusätzlich die gleitende
  Standardabweichung der Audio-Spalte über 30 s, je Aufnahme gerechnet.

Die neue Spalte wird VOR der Standardisierung angehängt, damit sie
dieselbe Behandlung bekommt wie jede andere.

## Was der Lauf NICHT beantwortet

AUC 0.73 sagt nichts darüber, ob die Spalte dem Kopf etwas HINZUFÜGT. Das
Backbone könnte dieselbe Information längst tragen — laute, statische,
schnittarme Bilder sehen anders aus als Dialogszenen. Genau diese
Redundanz misst der Lauf, und ein Null-Ergebnis hiesse „das Backbone
weiss es schon", nicht „das Audio ist wertlos".

## Rauschen, gemessen vor der Schwelle

Drei Seeds des Kontrollarms: 0.8981, 0.9060, 0.9022. Median 0.9022,
**sd 0.0039**.

## Bedingung

```regel
{
  "id": "O22",
  "frage": "Hebt die Lautheits-Schwankung als Zusatzspalte die binaere Leistung?",
  "art": "offline-kopf-ab",
  "nicht_in_serienabschluss": true,
  "metrik": "F1 auf test, geglaettet 10s je Aufnahme",
  "arme": {"kontrolle": "1282 Spalten", "versuch": "1283 = + gleitende sd(30s) der Audio-Spalte"},
  "paarung": "gleicher Seed, gleiche Zeilen, gleiche Architektur",
  "seeds": 5,
  "rauschen_sd_kontrollarm": 0.0039,
  "bedingungen": {
    "median_delta_f1_mindestens": 0.002,
    "positive_seeds_mindestens": 4
  },
  "konsequenz_bei_erfuellt": "Aenderung an der Merkmals-Extraktion VORSCHLAGEN (Go-Seite audio_rms.go + train-head.py). KEIN Deploy ohne Menschen-Entscheidung; die Aenderung entwertet den gesamten Merkmals-Cache und muss das wert sein.",
  "konsequenz_bei_verfehlt": "Audio-Spalte ist abgeschlossen: das Backbone traegt die Information bereits. Nicht ausbauen (Cache-Bruch), aber nie wieder anfassen."
}
```

Beide Bedingungen müssen gelten. Schwelle 0.002 ist rund die Hälfte der
ungepaarten Standardabweichung; die Paarung über den Seed nimmt einen
Teil heraus, die Vorzeichenbedingung trägt den Rest.

⚠️ **Der Preis eines Erfolgs ist hoch und steht vorher fest.** Eine neue
Spalte ändert die Eingabebreite von 1282 auf 1283. Damit ist **jede
gecachte Merkmalsdatei ungültig** und müsste neu extrahiert werden — und
737 von 1028 Aufnahmen haben keine Quelle mehr. Ein positives Ergebnis
ist deshalb ein VORSCHLAG, keine Umsetzung, und er konkurriert direkt mit
Idee 4 (antrainiertes Backbone), die denselben Preis hätte.

Anders als bei Idee 4 liesse sich die Spalte allerdings AUS DER
BESTEHENDEN Spalte 1281 rechnen, ohne die Quelle — die gleitende
Standardabweichung braucht nur die schon gespeicherten Werte. Das ist der
entscheidende Unterschied und der Grund, warum dieser Versuch ueberhaupt
lohnt.
