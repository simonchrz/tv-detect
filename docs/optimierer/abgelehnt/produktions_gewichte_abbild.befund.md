Die Klasse daten ist auf dem Trainingsteil ausgeschöpft: die Label-Hygiene der Produktion trifft im Abbild nur 389 von 506k Zeilen (0.08 %), und keine von sechs Daten-Varianten bewegt den Proxy über das Rauschen. Gemessen wird deshalb nur noch die Parität (Herkunft 2x + Hygiene) als Frage an die Abbild-Lücke.

Daten: Trainingsteil von K (opt_lauf.saetze/lade/baue_k, ohne O25), 527 Aufnahmen mit ads (merged 457, auto 61, user 9), 506k Zeilen, 1350 Spalten.
Lehrer: K-gleiches MLP (3 Köpfe, feste Seeds), in-sample auf dem ganzen Trainingsteil, wie der Champion als Lehrer der Produktion.
* Widerspruch > 0.9: auto 0.55 % der Zeilen, geprüft 0.60 %. Kein Unterschied zwischen Maschinen- und Menschenlabels.
* Hygiene wirft 389 Zeilen weg (y=1 128, y=0 261) aus 51 auto-Aufnahmen. Keine Aufnahme fällt unter den 30-%-Deckel.
* Die weggeworfenen Werbezeilen liegen zu 98 % unter sichtbarem Logo (>0.6). Die Hygiene trifft also den richtigen Posten, nur ist er im Abbild winzig.
Proxy: Hälften nach sha1(uuid), K-MLP mit allen Spalten, 2 Seeds. Bewertet auf geprüften Aufnahmen der anderen Hälfte: Sekundenfehler, FP/FN und 60-s-geglätteter Fehler je Aufnahme (Mittel/Median).
* Hygiene allein: err +0.0006 bis +0.0010 in 4/4 Läufen, FP +150 bis +250. Das Weglassen schwerer Sendungszeilen kostet mehr als das Weglassen falscher Werbezeilen bringt.
* Herkunft 2x: err −0.0009/+0.0001/−0.0002/+0.0005, e60-Median ±0.0004. Rauschen.
* Parität (Herkunft 2x × Hygiene): err −0.0010/+0.0009/−0.0002/+0.0009. Rauschen.
* Alle auto-Aufnahmen raus (auto_rest_ausschluss, nie gemessen): err −0.0023/−0.0002/+0.0001/+0.0005, FP −722 bis +90. Uneinheitlich.
* Serien-Ausgleich 1/sqrt(n Folgen) bzw. Deckel 5 Folgen (86 Titel, größte 48/47/45/43/38): err +0.0041/+0.0048/+0.0023/−0.0010. Schlechter.
* Aufnahmeränder ±5 min raus: FP +1500 bis +2800 in 4/4. Die Ränder lehren Sendung (FP liegen dort nur knapp doppelt so dicht: 0–120 s 3.3 % der Zeilen, 5–8 % der FP).
* ad-rate > 0.5 (Produktionsfilter): trifft im Trainingsteil 0 Aufnahmen. Leere Aufnahmen (44, alle auto-confirm-empty) lädt K nicht; die Produktion behandelt sie als Bootstrap.
* Gegenprobe Klasse nachbearbeitung (2-Zustands-Viterbi, Wechselstrafe 10/40): Logit-Spreizung z|z|/c, auch einseitig, verschiebt nur FP nach FN; der Aufnahme-Median steigt um 0.001–0.004. Gekappt (017) wirkt im Proxy weiter gut, im Messgerät aber −0.0032. Der Proxy taugt für diese Klasse nicht.
Erwartung: Δ ≈ 0 (±0.002). Wert des Laufs: klärt, ob Herkunft und Hygiene zur Abbild-Lücke (K 0.970 gegen Produktion 0.979) beitragen. Die Lehre aus 014: das Messgerät sah dort +0.005, wo der Proxy einen Verlust vorhersagte.
Skripte: Scratchpad hyg_diag.py, screen.py, pos.py, par.py, spreiz.py, spreiz2.py (read-only, nur Summen).

## Abgelehnt

ABLEHNEN, Kernfehler: Die Behauptung der Paritaet zur Produktion stimmt nicht. Der Arm setzt 'geprueft' als which in (user, merged) (GEPRUEFT, Zeilen 41, 95, 112). Die Nacht-Produktion laeuft aber seit 2026-09-06 mit --herkunft-belegt (daemon/tv-train-head.sh:256, auch tv-tagesserie.sh:143 und siglip-herausforderer.sh:49). Dort verlieren Aufnahmen mit mensch_belegt is False sowohl das 2x-Gewicht als auch die Ausnahme vom Hygiene-Veto (train-head.py:4243-4267, O17). Der Arm baut also den Zustand VOR O17 nach, also genau den Fehler, den die Produktion korrigiert hat. Im Archiv (erste 400 npz) tragen 74 von 304 merged-Eintraegen mensch_belegt=False (24 %), dazu 5 auto-confirm. Diese Aufnahmen bekommt der Arm 2x Gewicht und Hygiene-Schutz, die Produktion nicht. meta enthaelt 'mensch_belegt' (bei alten Eintraegen oft fehlend = None, dort behaelt die Produktion das Privileg); der Arm liest es nicht. Damit misst er weder die Produktionsgewichte noch die behauptete Luecke, und die Befund-Zahlen (geprueft 0.60 % Widerspruch, 51 auto-Aufnahmen betroffen) beruhen auf der falschen Aufnahmemenge.; Neuheit/Friedhof: README fuehrt 'Herkunfts-Gewichte (O17, O18)' als entschieden. Die Begruendung ('unveraenderte Vorgabe, keine neue Herkunftsregel') haelt nicht, weil die Produktionsregel gerade O17 ist (belegt) und der Arm eine andere, aeltere Regel einbaut. Das waere eine Herkunftsregel entgegen O17 und entwertet keine alte Messung, sondern widerspricht ihr. Die Altersrampe und die uebrige P;
