# Nachbefund 015 (17.10., nach dem Urteil, aendert es nicht)

Der Verlust sitzt fast ganz in einer Aufnahme: rtlzwei 1781935800 (Die
Geissens), IoU 0.942 -> 0.625 in allen 5 Wiederholungen gleich.

Replay der Arm-Ausgabe (Seed 4) durch den Produktions-HSMM:

| | Bloecke |
|---|---|
| Label | [0,75] [1870,2414] [3009,3513] |
| Produktion | [0,79] [1869,2416] [3058,3512] |
| Arm | — [1871,2412] [3065,3509] |

Der Arm verliert nur den ersten Block. Im Bild ist das **Teleshopping**
(Genius, feelvita-Kapseln): lange ruhige Einstellungen, Preistafeln, kaum
Schnitte. Die Bilddynamik liest das als verweilende Sendung -- genau gegen
die Annahme "Werbung schneidet schnell". Am Aufnahmebeginn ist das Fenster
zudem nur einseitig. Das Label ist richtig (Konvention: Dauerwerbung =
Werbung).

Die Bilddynamik selbst ist in dieser Aufnahme sauber: Fenster-Mittel
0.33-0.43 in beiden Werbebloecken, 0.14-0.30 in der Sendung.

Folgerung: Ein Nachfolger muss Teleshopping/Dauerwerbung anders tragen
(langsamer Schnitt ist dort kein Sendungshinweis). Ohne einen solchen
Mechanismus waere ein zweiter Anlauf Suchen auf dem Messsatz. Nicht
angesehen: super-rtl (-0.085, eine Aufnahme).
