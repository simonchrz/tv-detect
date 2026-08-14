# O9 — Kauft ein Ad-Bias in der HSMM-Emission die Trailer zurück?
(Vorab-Registrierung)

**Geschrieben 2026-08-14, VOR dem ersten Bias-Wert über den
Bestätigungssatz.** Bauart wie
[`hsmm-holdout-preregistration.md`](hsmm-holdout-preregistration.md):
Prosa mit Zahlen vorab, eingefrorene Sätze, entschiedene Ausgänge — kein
`regel`-Block, weil `audit-preregistration.py` nur Trainings-Serien aus
`shadow-trend.jsonl` lesen kann und dies ein deterministischer
Decoder-Sweep ohne Seeds und ohne Nächte ist.

## Was getestet wird

`--hsmm-ad-bias` (existiert samt Test in `internal/blocks/hsmm.go`:
`cumAd[i+1] = cumAd[i] + log(v)*EmitW + AdBiasLP`). **Positiv** = die
Emission neigt zu Werbung; Sekunden im Mittelband kippen von Sendung nach
Werbung, Blockkanten wandern nach außen.

## Warum überhaupt (§3x)

Über 19 Aufnahmen und 21 Trailer-Spannen trennt der Kopf sauber: Sendung
Median 0.009, Trailer 0.586, Werbung 0.921 — **21 von 21 Trailern über dem
90. Perzentil der Sendung.** 65 % der verpassten Werbung sind Trailer
(356 s auf dem Golden-Satz, 24 s/Std). Die Information liegt also im
Kopf-Ausgang; der Decoder verwirft sie. Eine handgeschnitzte
Randverlängerung hat das NICHT eingesammelt (beste Fassung −75 s von
1469 s Gesamtfehler, 16 Kanten besser gegen 12 schlechter — §3x). Der
Bias ist derselbe Gedanke am richtigen Ort: im Modell statt daneben.

## Kandidaten — abschließend, VOR der Messung

`0.0 (Referenz), 0.1, 0.2, 0.3, 0.5, 0.8, 1.2`

Sieben Werte. Mehr wird nicht probiert; wer nachträglich einen achten
braucht, hat die Frage verloren, nicht beantwortet.

## Sätze — eingefroren

**Abstimmung (22 Aufnahmen, NICHT im Golden-Satz, menschlich gelabelt,
frische Signale):** hier wird EIN Wert ausgewählt.

**Bestätigung (22 Aufnahmen = der Golden-Satz, soweit Material auf der
Pi liegt):** hier wird der ausgewählte Wert einmal gemessen. Kein
Nachjustieren, keine zweite Zelle.

⚠️ Der Bestätigungssatz IST der Maßstab. Ein Decoder-Parameter, der auf ihm
abgestimmt wird, macht ihn wertlos — deshalb die Trennung, und deshalb
wird die Abstimmung ausschließlich auf Aufnahmen gefahren, die nicht darin
liegen.

```json
{
 "abstimmung": [
  "dvr-kabel-eins-1783924200",
  "dvr-prosieben-1777383597",
  "dvr-prosieben-1778515939",
  "dvr-prosieben-1778559566",
  "dvr-prosieben-1778590189",
  "dvr-prosieben-1778655107",
  "dvr-prosieben-1778691925",
  "dvr-prosieben-1779111704",
  "dvr-prosieben-1781095127",
  "dvr-prosieben-1781520549",
  "dvr-prosieben-1782741830",
  "dvr-prosieben-1782828139",
  "dvr-prosieben-1783764997",
  "dvr-rtlzwei-1778523300",
  "dvr-rtlzwei-1779016500",
  "dvr-rtlzwei-1781935800",
  "dvr-rtlzwei-1781939400",
  "dvr-rtlzwei-1781943300",
  "dvr-rtlzwei-1781946900",
  "dvr-rtlzwei-1782645300",
  "dvr-vox-1778691600",
  "dvr-vox-1778778000"
 ],
 "bestaetigung": [
  "dvr-comedy-central-1784109600",
  "dvr-disney-channel-1779163800",
  "dvr-kabel-eins-1778511363",
  "dvr-nick-1781550300",
  "dvr-nick-1783889700",
  "dvr-prosieben-1778926191",
  "dvr-prosieben-1779089343",
  "dvr-prosieben-1779188536",
  "dvr-prosieben-1782450900",
  "dvr-prosieben-1782484228",
  "dvr-prosieben-1783401000",
  "dvr-rtl-1782090300",
  "dvr-rtlzwei-1779128100",
  "dvr-rtlzwei-1780341300",
  "dvr-rtlzwei-1784571300",
  "dvr-sat-1-1778572809",
  "dvr-sat-1-1781979300",
  "dvr-sixx-1779031634",
  "dvr-sixx-1780378004",
  "dvr-toggo-plus-1780640400",
  "dvr-vox-1778860800",
  "dvr-vox-1779206400"
 ]
}
```

## Messgröße

Block-IoU je Aufnahme gegen `ads_user.json`, über den **Produktionspfad**
(`tv-detect --replay-signals` mit `--decoder hsmm --hsmm-dur-w 15`), nicht
über eine Nachbildung in Python. Berichtet wird der Median über die
Aufnahmen, dazu die Zahl der Aufnahmen besser/schlechter und der größte
Einzelverlust.

## Hypothesen, mit Zahlen, vor den Daten

**H1 — es gibt überhaupt ein Optimum ungleich null.** Auf dem
Abstimmungssatz liegt der beste Bias-Wert bei ≥ 0.1 und verbessert den
Median-IoU um **≥ +0.005** gegenüber 0.0.

*Zuversicht: mittel.* Die Emissionsrechnung spricht dafür: bei v = 0.009
ist log(v) − log(1−v) ≈ −4.7, bei v = 0.586 bereits ≈ +0.35. Der Trailer
begünstigt Werbung also schon je Sekunde; was ihn kippt, ist der
Dauer-Prior. Ein kleiner Bias sollte genau dort greifen und die Sendung
bei 0.009 nicht erreichen — dafür bräuchte es ≈ +4.7.

**H2 — der ausgewählte Wert hält auf dem Bestätigungssatz.** Median-IoU
**> 0.0** gegenüber Bias 0.0, kein Einzelverlust größer **0.05** IoU, und
die Sekunden „Sendung weggeschnitten" (`fehlermoden.py`) steigen um
höchstens **20 %**.

*Zuversicht: gering bis mittel.* Zwei Gründe zu zweifeln, beide vorher
sichtbar: (1) Die Randverlängerung hat auf denselben Signalen fast genau
ausbalanciert (16:12) — der Fehler sitzt offenbar nicht nur dort, wo das
Mittelband ihn erklärt. (2) Ein GLOBALER Bias trifft jede Kante, auch die
schon richtigen; die Asymmetrie 356 s verpasst gegen 86 s zu großzügig ist
zwar deutlich, aber nicht so deutlich, dass ein pauschaler Schub sicher
netto gewinnt. **Ich erwarte H1 zu bestehen und H2 knapp zu scheitern.**

## Was jeder Ausgang bedeutet — jetzt entschieden

**H1 und H2 bestehen** → der Wert geht in die Produktions-Detect-Config
(per-Kanal-Override bleibt möglich, wird hier aber nicht mitgetestet). Neue
`label_hash`-Epoche ist NICHT betroffen; wohl aber ändert sich jede
ads.json, also läuft danach ein Nightly, bevor irgendeine Zahl mit früheren
verglichen wird.

**H1 besteht, H2 nicht** → der Bias ist ein Abstimmungs-Artefakt und
kommt in den Friedhof (§4). Die Trailer-Klasse bleibt offen und braucht
eine andere Idee als eine globale Verschiebung — dann ist der nächste
Kandidat eine KANTEN-lokale Emission (nur Sekunden in Reichweite einer
bestehenden Blockgrenze), nicht noch ein globaler Regler.

**H1 besteht nicht** → die Trailer sind für den Decoder unerreichbar,
obwohl der Kopf sie sieht; dann liegt es am Dauer-Prior und nicht an der
Emission. Auch das ist ein Ergebnis und beendet diese Spur.

⚠️ **Kein Ausgang rechtfertigt, den Golden-Boden zu senken oder Labels
anzufassen** (L1/L2).

---

# Ergebnis (2026-08-14, Abstimmungssatz)

```
ABSTIMMUNG (22 Aufnahmen)
  bias   Median    Delta  besser  schlechter  groesster Verlust
   0.0   0.9133  +0.0000       0           0              0.000
   0.1   0.9113  -0.0021       7           5             -0.046
   0.2   0.9113  -0.0021       6           9             -0.048
   0.3   0.9011  -0.0122       7           9             -0.109
   0.5   0.9136  +0.0003       6          11             -0.146
   0.8   0.9138  +0.0004       6          11             -0.184
   1.2   0.8986  -0.0147       7          12             -0.199
```

**H1 NICHT ERFÜLLT.** Gefordert waren ≥ +0.005 Median-IoU bei einem Wert
≥ 0.1. Der beste Wert bringt **+0.0004** — Rauschen — und erkauft ihn mit
6 besseren gegen 11 schlechtere Aufnahmen und einem Einzelverlust von
−0.184. Kein Kandidat kommt in die Nähe.

**Der Bestätigungssatz wurde NICHT angefasst.** Genau dafür war die
Trennung da; H2 wird nicht gemessen, weil H1 nichts auszuwählen übrig
lässt.

⚠️ **Meine Vorhersage war falsch.** Registriert stand „ich erwarte H1 zu
bestehen und H2 knapp zu scheitern" — begründet mit der Emissionsrechnung
(Trailer begünstigen je Sekunde bereits Werbung, Sendung liegt 4.7 Log
entfernt). Der Rechenweg stimmt, die Schlussfolgerung nicht: schon Bias 0.1
verschlechtert fünf Aufnahmen. Ein GLOBALER Regler trifft eben jede Kante,
und die Kanten, die schon richtig sitzen, sind in der Überzahl. Dieselbe
Lehre wie bei der Randverlängerung in §3x — dort 16:12, hier 6:11.

**Konsequenz, wie vorab entschieden:** die Emission ist nicht der Ort. Wenn
die Trailer überhaupt decoder-seitig erreichbar sind, dann über den
DAUER-Prior oder eine kanten-lokale Emission — beides eigene Fragen mit
eigener Registrierung, nicht als Nachschlag hier.
