> **ABGESCHLOSSEN 2026-09-25 — REGEL ERFÜLLT.** Median-ΔF1 **+0.0313**
> (test mit Spur), **5 von 5** Seeds positiv (+0.022 / +0.032 / +0.023 /
> +0.033 / +0.031); Schwelle +0.004 und 4/5. Kontrollarm reproduziert die
> Vorabmessung Wert fuer Wert (Median 0.8815). Nebenwert alle test +0.0031
> (nur 31 % der test-Zeilen haben eine Spur). Ergebnis:
> `~/.cache/tvd-train-archive/o26-ergebnis.json`, Log `~/Library/Logs/o26-lauf.log`.
> Konsequenz laut Registrierung: Produktionsweg VORSCHLAGEN, nicht bauen —
> vorher die Kosten im Detect messen.

# O26 — Hebt OCR (Bildschirm-Text) als Zusatzspalte die binäre Leistung? (Vorab-Registrierung)

**Geschrieben 2026-09-25, vor dem ersten Behandlungs-Datenpunkt.** Bauart wie
[`o22-audio-dynamik-preregistration.md`](o22-audio-dynamik-preregistration.md):
offline, gleiche Zeilen, gleiche Seeds, gleiche Architektur; die Arme
unterscheiden sich NUR in drei angehängten Spalten. Skript:
`scripts/o26-ocr-spalte.py`, Test `scripts/test_o26_ocr_spalte.py`.

## Woher die Frage kommt

Der Backbone bekommt 224×224 und kann eingeblendeten Text nicht lesen
(Memory `backbone_liest_keinen_text`, Trailer sind die größte Fehlerklasse,
Überanpassungslücke train/test +0.251). Die flächendeckende OCR-Spur
(`tv-ocr-spur`, 2026-09-24, 322 Aufnahmen, Messsatz 98/98) holt die
verworfene Information zurück — sie rechnet nichts um, was schon da ist
(die Sackgasse aller früheren Zusatzspalten, O2/O6/O7/O8/O16).

Gemessen am Messsatz, OHNE Training (2026-09-25): NN-Fehlersekunden liegen
5,5-mal so oft wie richtige in ±10 s eines OCR-Treffers; nach Abstand zur
Label-Kante getrennt bleibt die Anreicherung (0–30 s 1,6×, 30–90 s 2,5×,
90–300 s 5,1×, >300 s 2,6×). Das ist ein Zusammenhang. Ob ein Kopf ihn
nutzen kann, misst diese Frage.

## Die Spalten (je Sekunde, hinten angehängt)

| Spalte | Wert |
|---|---|
| `hinweis_nah` | 1, wenn ein Programmhinweis (Wochentag + Uhrzeit) in ±10 s liegt |
| `werbung_nah` | 1, wenn eine Werbe-Kennzeichnung („Werbung“) in ±10 s liegt |
| `spur_da` | 1, wenn die Aufnahme eine Spur hat und die Sekunde abgetastet ist |

Ohne `spur_da` wäre „kein Text“ von „nie hingesehen“ nicht zu trennen; 71 %
der train-Aufnahmen haben keine Spur (Quelle nicht mehr vorhanden).

## Rauschen, VOR der Behandlung gemessen

Nur Kontrollarm, 5 Seeds (2026-09-25, `--nur-kontrollarm`): F1 test mit Spur
0.8806 / 0.8745 / 0.8838 / 0.8850 / 0.8815, **Median 0.8815, sd 0.0041**.
Geladen: train 671 Aufnahmen (205 mit Spur), test 142 (46 mit Spur).

## Regel

```regel
{
  "id": "O26",
  "frage": "Hebt OCR (Bildschirm-Text) als Zusatzspalte die binaere Leistung?",
  "art": "offline-kopf-ab",
  "nicht_in_serienabschluss": true,
  "metrik": "F1 auf den test-Aufnahmen MIT OCR-Spur, geglaettet 10s je Aufnahme (Nebenwert: alle test-Aufnahmen)",
  "arme": {"kontrolle": "1282 Spalten", "versuch": "1285 = + hinweis_nah, werbung_nah, spur_da (+-10 s)"},
  "paarung": "gleicher Seed, gleiche Zeilen, gleiche Architektur",
  "seeds": 5,
  "rauschen_sd_kontrollarm": 0.0041,
  "bedingungen": {
    "median_delta_f1_mindestens": 0.004,
    "positive_seeds_mindestens": 4
  },
  "konsequenz_bei_erfuellt": "Produktionsweg VORSCHLAGEN, nicht bauen: flaechendeckende OCR im Detect (~1 s je Minute Video) + Spalten im Nightly (L5: Header-Bump, Paritaets-Fixture, ausdrueckliches OK). Vorher die Kosten im Detect messen.",
  "konsequenz_bei_verfehlt": "OCR als Kopf-Spalte ist fuer jetzt erledigt. Die Spur bleibt als Werkzeug (Label-Audit, Trailer-Zonen), nicht als Eingang."
}
```

## Warum die Schwelle strenger ist als bei O22

O22 verlangte +0.002 (≈0,5 sd). Hier +0.004 (≈1 sd): ein Erfolg hätte
echte Kosten — flächendeckende OCR in jedem Detect und eine Header-
Migration. Ein Effekt, der das nicht deutlich trägt, soll als verfehlt
gelten. Eine Serie mit Vorzeichen, aber unter der Schwelle, ist verfehlt.

## Was diese Frage NICHT beantwortet

* Nicht den Golden-Median: dort haben nur 23 von 38 eine Spur.
* Nicht, ob OCR als DEKODER-Evidenz trägt (O10 scheiterte dort an der
  Referenz, O13 sammelt kanten-lokal).
* Nicht den Produktionsfall mit voller Abdeckung: im Training haben nur
  205 von 671 Aufnahmen eine Spur. Ein Erfolg hier ist eine Untergrenze.
