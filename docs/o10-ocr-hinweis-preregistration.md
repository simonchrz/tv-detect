# O10 — Kauft ein OCR-Programmhinweis am Blockrand die Trailer zurück?
(Vorab-Registrierung)

**Geschrieben 2026-08-14, VOR der ersten Regel-Messung.** Bauart wie
[`o9-adbias-preregistration.md`](o9-adbias-preregistration.md), gleiche
eingefrorene Sätze, gleiche Trennung Abstimmung/Bestätigung.

## Was getestet wird

Eine **kanten-lokale** Regel — genau der Kandidat, den O9 für den Fall
ihres Scheiterns benannt hat, jetzt mit Inhalt: nach dem hsmm-Blockende
vorwärts nach einem **Programmhinweis** suchen (OCR-Muster: Wochentag /
„Heute" / „MO-FR" / „Jeden Tag" **und** eine Uhrzeit) und den Block bis
hinter den letzten Treffer verlängern.

Kein globaler Regler. Die Regel fasst ausschließlich Sekunden in Reichweite
einer bestehenden Blockgrenze an, erzeugt nie einen Block und verkürzt nie
einen.

## Warum (§3aa)

Die Überanpassungs-Lücke train↔test ist bei Trailern +0.251 gegen +0.023
(Sendung) und +0.007 (Werbung-Kern): der Kopf merkt sie sich, statt sie zu
lernen. Er kann nicht anders — der Backbone bekommt 224×224 aus 1920×1080,
die Programmhinweis-Leiste ist darin ~8 px hoch. OCR holt Information
zurück, die der Eingang physisch verwirft; das ist kategorisch etwas
anderes als die abgeleiteten Spalten aus §3w.

## ⚠️ Was an dieser Registrierung SCHWÄCHER ist als bei O9

Das OCR-**Muster** wurde an 11 Trailer-Spannen entworfen, von denen einige
in Aufnahmen des Bestätigungssatzes liegen. Der Bestätigungssatz ist über
das Muster also nicht naiv. Naiv ist er über die **Regel-Parameter** und
über die eigentliche Frage — ob eine Blockverlängerung daraus den IoU hebt,
ist eine andere Größe als ob das Muster Trailer erkennt.

Ich schreibe das hin, statt es zu überspielen: ein bestandenes O10 belegt
„die Regel trägt auf ungesehenen Parametern", nicht „das Muster wurde
unabhängig bestätigt". Für Letzteres bräuchte es Trailer-Spannen, die
niemand beim Musterentwurf gesehen hat.

## Kandidaten — abschließend, VOR der Messung

* Suchfenster nach dem Blockende: **60, 90, 120 s**
* Abtastung: alle **2 s** (fest, nicht variiert)
* Nachlauf hinter dem letzten Treffer: **5, 10, 15 s**

Neun Zellen. Nur **Blockenden**; die Startseite wird berichtet, aber NICHT
zur Auswahl herangezogen — sie war in §3x und §3z in jeder Zelle negativ.
Wer nachträglich eine zehnte Zelle braucht, hat die Frage verloren.

## Sätze — eingefroren (identisch zu O9)

Abstimmung 22 Aufnahmen (nicht im Golden-Satz), Bestätigung
22 Aufnahmen (= Golden-Satz mit Material). Alle 44 haben eine
lokale Quelle, die Frames kommen also aus dem Cache und nicht über HLS.

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

Block-IoU je Aufnahme gegen `ads_user.json`. Ausgangspunkt sind die Blöcke
des Produktionspfads (`tv-detect --replay-signals`, `--decoder hsmm
--hsmm-dur-w 15`) aus den frischen Signal-Dumps; die Regel wird darauf
angewandt. Berichtet: Median, Aufnahmen besser/schlechter, größter
Einzelverlust.

## Hypothesen, mit Zahlen, vor den Daten

**H1 — auf dem Abstimmungssatz.** Die beste der neun Zellen verbessert den
Median-IoU um **≥ +0.005** gegenüber „Regel aus", bei **mehr besseren als
schlechteren** Aufnahmen.

*Zuversicht: mittel bis hoch.* Der Trennwert ist außergewöhnlich sauber
(10/11 gegen 0/32 gegen 0/120), und die Regel greift nur dort, wo ohnehin
eine Kante liegt. Das ist ein deutlich besseres Ausgangsblatt als bei O9.
Zwei Gründe zur Vorsicht bleiben: der Programmhinweis erscheint oft nur in
einem Teil des Trailers, der Nachlauf muss also die Lücke bis zum echten
Sendungsbeginn überbrücken — und wo der Trailer VOR dem Blockende endet,
verlängert die Regel ins Leere.

**H2 — auf dem Bestätigungssatz.** Derselbe Wert: Median-IoU **> 0**, kein
Einzelverlust größer **0.05**, „Sendung weggeschnitten" (`fehlermoden.py`)
steigt um höchstens **20 %**.

*Zuversicht: mittel.* Vorhersage: **H1 besteht, H2 besteht knapp.** Bei O9
lag ich mit „H1 besteht" falsch; der Unterschied ist, dass dort ein
globaler Regler jede Kante traf, während hier nichts passiert, wo kein
Programmhinweis steht.

## Was jeder Ausgang bedeutet — jetzt entschieden

**Beide bestehen** → die Regel geht in `tv-detect` (kanten-lokal, mit
`--ocr-hinweis`-Schalter, standardmäßig AUS bis ein Nightly sie bestätigt)
und OCR wird kanten-lokal in den Detect-Lauf gehängt (+17 %). Danach erst
lohnt die teure Variante: eigene Spalte, Neu-Extraktion des Korpus.

**H1 besteht, H2 nicht** → Friedhof (§4). Dann ist die Trailer-Klasse
decoder-seitig auf DREI Wegen gescheitert, und die Konsequenz ist die
teure: das Merkmal gehört in den Kopf, nicht hinter ihn.

**H1 besteht nicht** → das Muster erkennt Trailer, taugt aber nicht zum
Setzen von Kanten. Auch dann bleibt der OCR-Befund (§3aa) gültig — er
zeigte nie, dass die KANTE dort liegt, wo der Text steht.

⚠️ Kein Ausgang rechtfertigt, den Golden-Boden zu senken oder Labels
anzufassen (L1/L2).
