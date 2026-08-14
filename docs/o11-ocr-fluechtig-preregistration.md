# O11 — Trägt der OCR-Programmhinweis, wenn Dauer-Einblendungen
ausgeschlossen sind? (Vorab-Registrierung)

**Geschrieben 2026-08-14, VOR der ersten Messung auf dem neuen Satz.**
Nachfolger von [`o10-ocr-hinweis-preregistration.md`](o10-ocr-hinweis-preregistration.md),
die an H1 scheiterte (3 bessere gegen 4 schlechtere Aufnahmen).

## Was sich gegenüber O10 ändert — und was NICHT

O10s vier Verlierer waren per Frame-Review **3× Trailer** (die Labels waren
zu kurz; korrigiert) und **1× Dauer-Einblendung**: bei
`dvr-rtlzwei-1779016500` klebt „NEUE FOLGE HEUTE 20:15 / KAMPF DER REALITY
ALLSTARS" oben rechts durch die ganze Sendung und löst das Muster
dauerhaft aus. Diese eine Aufnahme trug den größten Einzelverlust (−0.144).

**Neu ist genau eine Bedingung — Flüchtigkeit.** Ein Treffer zählt nur,
wenn dasselbe Muster im Referenzfenster **+180 s bis +240 s** nach dem
Blockende NICHT mehr steht. Ein Trailer ist 10–30 s lang, eine
Dauer-Einblendung steht Minuten.

⚠️ **Warum ein neuer Satz sein MUSS:** O10s Abstimmungssatz taugt nicht
mehr als Referenz — ich habe drei seiner Labels selbst korrigiert, und
zwar zugunsten der Regel. Dieselbe Regel darauf erneut zu messen wäre
zirkulär. Deshalb 60 Aufnahmen, die an O9 und O10 unbeteiligt waren.

## Was ABSICHTLICH aus O10 übernommen wird — offengelegt

Suchfenster **60 s** und Nachlauf **5 s** sind nicht neu abgestimmt,
sondern aus O10s bester Zelle übernommen. Das ist datengetrieben und wird
hier hingeschrieben statt versteckt: es verkleinert den Suchraum auf die
EINE neue Idee, statt neun Zellen ein zweites Mal zu durchsuchen.

## Kandidaten — abschließend

Zwei Zellen:

1. **Regel aus** (Referenz)
2. **Regel mit Flüchtigkeits-Bedingung**

Zusätzlich wird **Regel ohne Flüchtigkeit** berichtet — nur, um sichtbar
zu machen, was die Bedingung kauft. Sie ist NICHT auswählbar.

Nur Blockenden. Keine dritte Zelle, kein Nachjustieren.

## Sätze — eingefroren

**Abstimmung: 60 Aufnahmen**, unbeteiligt an O9/O10, menschlich gelabelt,
lokale Quelle, stratifiziert über 11 Kanäle (höchstens 12 je Kanal, damit
prosieben den Satz nicht trägt).

**Bestätigung: 22 Aufnahmen** = der Golden-Satz mit Material,
unverändert aus O9/O10 und dort nie gemessen.

⚠️ Dieselbe offengelegte Schwäche wie bei O10: das OCR-**Muster** wurde an
Trailer-Spannen entworfen, von denen einige in Golden-Aufnahmen liegen. Der
Bestätigungssatz ist über das Muster nicht naiv — wohl aber über die
Flüchtigkeits-Bedingung und über die Frage, ob daraus bessere Kanten werden.

```json
{
 "abstimmung": [
  "dvr-comedy-central-1778837700",
  "dvr-comedy-central-1780940400",
  "dvr-comedy-central-1781026800",
  "dvr-disney-channel-1779140400",
  "dvr-disney-channel-1779225300",
  "dvr-disney-channel-1781552400",
  "dvr-disney-channel-1781554200",
  "dvr-kabel-eins-1778597792",
  "dvr-kabel-eins-1779119815",
  "dvr-kabel-eins-1779375379",
  "dvr-kabel-eins-1779379007",
  "dvr-kabel-eins-1779810974",
  "dvr-nick-1778578800",
  "dvr-nick-1778775300",
  "dvr-nick-1778860200",
  "dvr-nick-1781292600",
  "dvr-nick-1781374500",
  "dvr-nick-1781721300",
  "dvr-nick-1782930900",
  "dvr-prosieben-1778676632",
  "dvr-prosieben-1778766521",
  "dvr-prosieben-1778778286",
  "dvr-prosieben-1779194893",
  "dvr-prosieben-1779887700",
  "dvr-prosieben-1780545300",
  "dvr-prosieben-1780847847",
  "dvr-prosieben-1781267782",
  "dvr-rtl-1780078500",
  "dvr-rtl-1780990200",
  "dvr-rtl-1781026800",
  "dvr-rtl-1781113200",
  "dvr-rtl-1781545200",
  "dvr-rtl-1782111600",
  "dvr-rtl-1785678300",
  "dvr-rtlzwei-1779131700",
  "dvr-rtlzwei-1780946100",
  "dvr-rtlzwei-1781550900",
  "dvr-rtlzwei-1782044100",
  "dvr-rtlzwei-1783977300",
  "dvr-rtlzwei-1784574900",
  "dvr-rtlzwei-1784578500",
  "dvr-rtlzwei-1784582400",
  "dvr-sat-1-1779112822",
  "dvr-sat-1-1779978600",
  "dvr-sat-1-1780065000",
  "dvr-sat-1-gold-1779170703",
  "dvr-sat-1-gold-1779257385",
  "dvr-sat-1-gold-1779508422",
  "dvr-sixx-1778660923",
  "dvr-sixx-1778917321",
  "dvr-sixx-1779122013",
  "dvr-sixx-1780411200",
  "dvr-sixx-1780500900",
  "dvr-vox-1778943000",
  "dvr-vox-1778946900",
  "dvr-vox-1779037800",
  "dvr-vox-1779292500",
  "dvr-vox-1785085800",
  "dvr-vox-1785517200",
  "dvr-vox-1786550400"
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

## Hypothesen, mit Zahlen, vor den Daten

**H1 — Abstimmungssatz.** Die Regel mit Flüchtigkeits-Bedingung verbessert
den Median-IoU um **≥ +0.005** gegenüber „Regel aus", **und** es sind mehr
Aufnahmen besser als schlechter, **und** kein Einzelverlust größer **0.10**.

*Zuversicht: mittel.* Dafür spricht die Diagnose: von O10s neun Treffern
waren nach Frame-Review acht richtig und einer eine Dauer-Einblendung —
die Bedingung entfernt genau den einen. Dagegen spricht, dass nur ~9 von 40
Blockenden überhaupt einen Hinweis trugen; bei 60 Aufnahmen sind ~20-25
Treffer zu erwarten, und der Median über 60 Aufnahmen bewegt sich davon
weniger als über 22.

**H2 — Bestätigungssatz.** Derselbe Aufbau: Median-IoU **> 0**, kein
Einzelverlust größer **0.05**, „Sendung weggeschnitten" steigt um höchstens
**20 %**.

**Vorhersage: H1 besteht, H2 besteht.** Bei O9 lag ich mit „H1 besteht"
falsch, bei O10 mit „H1 besteht" ebenfalls (knapp, an der zweiten
Bedingung). Ich sage es trotzdem vorher, weil eine unausgesprochene
Erwartung nicht widerlegbar ist.

## ⚠️ Was diesmal AUSGESCHLOSSEN ist

**Verlierer werden nicht per Frame-Review „gerettet".** Bei O10 war das
legitim, weil es der erste Blick auf den Fehlermodus war und die Regel
trotzdem als gescheitert im Protokoll steht. Ein zweites Mal wäre es ein
Muster: so lange an der Referenz drehen, bis die eigene Regel gewinnt. Wenn
O11 verliert, verliert O11. Ein Frame-Review der Verlierer darf danach
stattfinden — als Label-Arbeit, nicht als Nachverhandlung, und das Ergebnis
ändert O11s Urteil nicht.

## Was jeder Ausgang bedeutet — jetzt entschieden

**Beide bestehen** → OCR kanten-lokal in den Detect-Lauf (+17 %), Regel
hinter `--ocr-hinweis` (standardmäßig AUS bis ein Nightly bestätigt).

**H1 besteht, H2 nicht** → Friedhof. Dann ist der vierte Decoder-Weg
gescheitert und die Konsequenz die teure: das Merkmal gehört in den Kopf
(eigene Spalte, Neu-Extraktion), nicht dahinter.

**H1 besteht nicht** → das Muster erkennt Trailer zuverlässig (§3aa steht
unabhängig davon), taugt aber nicht zum Setzen von Kanten. Damit ist die
kanten-lokale Spur zu Ende und nur noch der Kopf-Weg offen.

⚠️ Kein Ausgang rechtfertigt, den Golden-Boden zu senken (L1) oder Labels
des Bestätigungssatzes anzufassen (L2).

---

# Ergebnis (2026-08-14, Abstimmungssatz)

56 Aufnahmen auswertbar, 101 Blockenden, **13 mit Hinweis, davon 1
dauerhaft** (die Flüchtigkeits-Bedingung greift also genau einmal).
Basis-Median 0.9658.

```
                             Zelle   Median    Delta  besser  schlechter  Verlust
           Regel MIT Fluechtigkeit   0.9686  +0.0028       7           3   -0.049
            Regel ohne (Kontrolle)   0.9686  +0.0028       8           3   -0.049
```

**H1 NICHT ERFÜLLT.** Zwei der drei Bedingungen sind erfüllt — mehr bessere
als schlechtere Aufnahmen (7:3) und kein Einzelverlust über 0.10 (−0.049).
Die erste ist es nicht: **+0.0028 statt der geforderten +0.005.**

**Der Bestätigungssatz wurde NICHT angefasst.**

## ⚠️ Der Fehler steckt in MEINER Registrierung, nicht im Ergebnis

Der Median über ALLE 56 Aufnahmen ist für einen **spärlichen** Eingriff die
falsche Kennzahl: die Regel fasst nur 11 Aufnahmen an, 45 bleiben
unverändert — der Median sitzt also mitten in den Unberührten und kann sich
kaum bewegen. Diagnose auf den 11 tatsächlich angefassten Aufnahmen:

```
  Median-Delta +0.0201   Mittel +0.0415
  besser 7   unveraendert 1   schlechter 3
  Spanne [-0.049 .. +0.190]
```

Das ist ein deutlicher Effekt — er war nur an einer Kennzahl gemessen, die
ihn strukturell nicht sehen kann. Die Schwelle +0.005 stammt aus O9, wo ein
GLOBALER Regler jede Aufnahme anfasste; sie unbesehen auf einen
kanten-lokalen Eingriff zu übertragen war mein Fehler.

**Das ändert das Urteil nicht.** Die Registrierung existiert genau dafür,
dass eine schlecht gewählte Kennzahl nicht nachträglich gegen eine bessere
getauscht wird, sobald das Ergebnis vorliegt.

## Wo das jetzt steht — und die Gefahr, die ich benenne

Drei Registrierungen, drei Fehlschläge, drei verschiedene Gründe: O9 an der
Sache (globaler Regler taugt nicht), O10 an der Referenz (Labels waren
kurz), O11 an der Kennzahl (Median blind für spärliche Eingriffe). Jeder
Grund war echt und ist protokolliert — aber **das Muster selbst ist ein
Warnsignal**: wer oft genug neu registriert, gewinnt irgendwann durch
Zufall.

Deshalb keine vierte Abstimmungsrunde. Was bleibt, ist ein Kandidat mit
plausibler Wirkung (+0.020 Median auf den angefassten Aufnahmen, 7:3) und
genau eine unverbrauchte Instanz, die ihn entscheiden kann: der
Bestätigungssatz. Der nächste Schritt ist **ein einziger Schuss darauf**,
mit vorab festgelegter, sparsamkeits-tauglicher Bedingung — und das
Ergebnis gilt, in welche Richtung auch immer.
