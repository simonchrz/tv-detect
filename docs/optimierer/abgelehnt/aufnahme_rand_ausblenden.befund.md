Klasse labels gibt auch mit drei neuen Pruefungen nichts her (Trainingslabels folgen den Konventionen mindestens so streng wie die Entwicklung); in daten bleibt die Rand-Konvention: die ersten/letzten 60 s jeder Aufnahme sind doppelt so fehlertraechtig und werden hier nicht gelernt.

Daten: saetze()/lade() aus opt_lauf (ohne O25), Trainingsteil 527 Aufn. mit ads, jede 4. s, 506k Zeilen; OCR-Spur ~/.cache/tvd-ocr-spur (nur gelesen). Skripte Scratchpad r1-r10b.py.
Labels (verworfen):
* OCR-Treffer-Cluster (Luecke <= 20 s, >= 2 Treffer, ab 120 s) bis 30 s an einer Kante: Anteil INNERHALB des Labels
  training werbemarker 386/402 = 0.960, hinweis 310/326 = 0.951; entwicklung (Summe) 0.911 / 0.921.
  -> Training folgt der Rand-Konvention schon staerker als die Entwicklung; rand_einblendung_ziel (nie gemessen) haette nichts zu tun.
  (Die 80 % im Befund rand_einblendung_ziel kamen aus den +-10-s-gespreizten Spalten, nicht aus Clustern.)
* Isolierte Countdown-/Einzelspots (OCR werbemarker > 30 s von jedem Block): training 70 Cluster Mitte (meist 8-32 s, merged 53 / auto 17);
  entwicklung hat ebenso 16 solche Cluster ausserhalb der Menschenbloecke und KEINEN kurzen Block (< 120 s) mit OCR-Cluster
  -> gleiche Konvention wie Training, plus Inselsperre 90 s im Dekoder: Umlabeln wuerde nichts oder Schaden messen.
* Nachlauf (letzter Block bis ans Ende, Logo-Mittel > 0.6): training nur 3 Bloecke / 931 s, entwicklung 2 / 534 s. Zu klein.
* confirmed_ad_skips fern (> 30 s) von Labelbloecken: 33 von 175 Punkten in 21 Aufn. Zu klein, Ausdehnung unbekannt.
Daten (Pruefungen):
* Bildstoerung: Off-Manifold-Zeilen (PCA64-Residuum, robuster z je Aufnahme) sind Werbung, nicht Stoerung: p99 Werbeanteil 0.59,
  nur 2 Laeufe >= 20 s -> der Trainingsteil enthaelt praktisch keine gestoerten Strecken (wie 032); Robustheit geht nur per Augmentation (training/merkmale).
* Lokaler Zeitachsen-Bruch mitten in der Aufnahme (Kanten-Versatz per Logo, Praefix ~0, Suffix >= 8 s): nur 3 Aufn. -> nichts.
* Raender (Proxy: Logistik auf Backbone-PCA64+Logo+Audio, Haelften nach sha1(uuid), balanced):
  Fehler gesamt 0.116; erste 60 s 0.145 (Werbeanteil 0.024), letzte 60 s 0.224 (0.088), letzte 180 s 0.152, Mitte 0.113.
  Raender je 60 s = 3.1 % der Zeilen, 6.1 % der erfundenen Werbung (FP), 1.3 % der verpassten.
Arm: Gewicht 0 fuer sek < 60 und die letzten 60 s; nur Trainingskopie. Risiko: K sagt am Rand der Eval-Aufnahmen
eher Werbung, wo die Menschenlabels (gleiche Konvention) Sendung sagen -> nn_erfindet am Rand kann steigen.

## Abgelehnt

ABLEHNEN wegen Neuheit (Pruefpunkt 3): Dieselbe Idee steht schon im Friedhof, in /Users/simon/src/tv-detect/docs/optimierer/abgelehnt/spielfilm_aufnahmen_gewicht.befund.md unter 'Verworfen vor dem Arm (Klasse daten)', Punkt 'Randzonen-Konvention'. Dort steht: 'Sendung-Zeilen in den ersten/letzten 120 s auf Gewicht 0: Fehler +0.0021 (4/4 schlechter), auch in der Mitte +0.0012; 300 s: +0.006. Tot.' Der neue Arm ist eine Parametervariante davon: 60 s statt 120 s, Gewicht 0 auf alle Zeilen statt nur auf Sendung-Zeilen. Die Begruendung nennt diesen Eintrag nicht. Die Liste 'Warum nicht im Friedhof' fuehrt ihn nicht auf. Sie erklaert auch nicht, warum die neue Messung etwas anderes misst, obwohl die frueheren Messungen gerade in der Mitte schlechter wurden. Die README verlangt fuer eine Variante eines verworfenen Arms eine solche Erklaerung.

Der Befund nennt das Risiko selbst: K bekommt an den Raendern der Eval-Aufnahmen keine Trainingsdaten mehr und sagt dort eher Werbung, waehrend die Menschenlabels dort Sendung sagen. Dann steigt nn_erfindet am Rand. Gewicht 0 auf ALLE Randzeilen entfernt ausserdem die korrekt gelabelten Werbe-Endbloecke. Ledger 5159ff.: 42 von 54 Endbloecken sind echte Werbung (2.3 h), nur 12 sind Folgesendung. Die Rand-Inkonsistenz ist laut Ledger nur etwa 0.25 % des Korpus gross (Werbung am Aufnahmerand 0.9 h in 100 Aufnahmen bei 628 h). Der Beleg des Arms (Proxy-Fehlerquote 0.224 in den letzten 60 s) trennt falsche Labels nicht von echter Schwierigkeit am Rand.
