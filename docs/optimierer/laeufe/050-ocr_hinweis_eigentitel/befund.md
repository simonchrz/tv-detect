ocr_hinweis wirft zwei gegensaetzlich gelabelte Formen zusammen: Programmhinweise auf FREMDE Titel sind zu 93 % Werbung (Trailer), solche auf die EIGENE Sendung nur zu 21 % (Bauchbinde/Eigenwerbung); getrennt halbiert das die OOF-Fehler in dieser Zone, die aber klein ist.

Klasse merkmale (Bandit, Zug 0.48). Daten: Trainingsteil saetze(), 191 von 665 Aufnahmen mit gueltiger OCR-Spur (spur_passt), Entwicklung 76/76.
OCR-Funde (nur geflaggte Frames) gegen meta-ads, Titelwort-Abgleich (Woerter >= 4 Zeichen, Stoppliste):
* Hinweis ohne Werbemarker: fremder Titel n=7708 Funde 0.93 Werbung, eigener Titel n=505 0.21.
* Werbemarker ohne Hinweis: fremd 0.84, eigen 0.58 (nicht angefasst: Countdown-Spots "Werbung - gleich geht's weiter GALILEO" sind Werbung).
* Hinweis+Werbemarker: 0.95, Titel egal.
Sekunden (+-10-s-Fenster wie ocr_spalten.py): Training nur-eigen 2180 s (23 % Werbung), fremd 38792 s (85 %); Entwicklung nur-eigen 630 s (Summe, ~8 s je Aufnahme).
Gerichtete Fenster (fremde Karte in den naechsten 11-40 s) neu geprueft: logo >= 0.5 nur 0.22 Werbung, logo < 0.5 0.88 -- das ist 004 (ocr_logo_fenster, -0.0023), nicht wiederholt.
OOF-Proxy: K-Spalten (baue_k), MLP 96 / 12 Ep. / 3er-Logit-Mittel, Haelften nach sha1(uuid), Schwelle Logit 0, Zeilenfehler (jede 4. s, 506k Zeilen):
* seed0: K 34287 -> Arm 33876 (-1.2 %); Zone nur-eigen 97 -> 45 von 556 (FP 95 -> 24); Hinweis-Zone 823 -> 766.
* seed1: K 34791 -> Arm 34665 (-0.4 %); Zone 95 -> 38 (FP 94 -> 25); Hinweis-Zone 835 -> 786.
Wirkung in der Zone eindeutig, Zone aber nur 0.1 % der Zeilen; der Gesamtgewinn ist zum Teil Rauschen. Erwartung realistisch +0.000..+0.001 Block-IoU.
Skripte: Scratchpad d/ocrtext.py, sek.py, vor.py, oof.py (nur Summen).
