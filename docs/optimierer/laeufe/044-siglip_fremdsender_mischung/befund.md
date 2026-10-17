Decode-Fehler erreichen die Merkmale nicht (super-rtl-1781112000 ist im Backbone/SigLIP unauffaellig); auffaellig ist die Abdeckung: Kinder-/Nischensender kennt K fast nur ohne SigLIP-Spur, und dort halbiert ein Logit-Mittel mit dem spurlosen Modus die erfundene Werbung im Proxy.

Klasse labels gezogen (Zug 0.89, 0 Laeufe), zwoelfter Anlauf ohne Hebel, daher merkmale (Zug 0.54):
* Rand-OCR-Konvention (Arm rand_einblendung_ziel, nie gelaufen): Kanten mit OCR-Werbung/Hinweis-Lauf ausserhalb des Labels
  (32 s, Luecke 4 s) training 466/750 (62 %), Entwicklung (Mensch) 145/254 (57 %) -> Menschenlabels folgen derselben Lesart, kein Labelhebel.
* Kurze Luecken/Bloecke: training 4 Luecken < 30 s (70 s gesamt), Entwicklung 0 -> sauber.
Sitzungshinweis Decode-Stoerung (Archiv-Features, 1-s-Raster, Skripte Scratchpad st/d1-d4.py):
* Backbone-Flackern (cos(t-1,t+1) - max Nachbar-cos > 0.3): training (98 Aufn.) Median 0.93 %, p95 2.5 %; Entwicklung 0.79 %/2.8 %;
  super-rtl 1.4 %. 60-s-Fenster > 20 % Flackern: nirgends. Norm |z| > 6: super-rtl 0. Standbild (cos > .97) super-rtl 0.9 % (training p5 0.8 %).
* SigLIP-Nachbarkosinus < 0.5: training Median 24 %, max 34 %; super-rtl 31 %. -> eine Stoerungsspalte wuerde dort NICHT anschlagen,
  Rauschen-Augmentation haette kein Gegenstueck (deckt sich mit 042). Einzige Auffaelligkeit super-rtl: Logo 0 ueber 99.6 % (Logo-Arme durch).
Abdeckung (siglip_da je Aufnahme, Sender aus meta/uuid): training mit Spur vox 39, prosieben 49, kabel-eins 35, rtlzwei 11, rtl 8,
  disney 5, sixx 5, nick 3, comedy-central 3, sat-1 3, sat-1-gold 3, super-rtl/toggo 0. Entwicklung 76/76 mit Spur, davon 24 auf diesen
  Sendern (nick 7, sixx 5, comedy 4, disney 4, sat-1 2, sat-1-gold 1, super-rtl 1); Bestaetigung 7/22 (inkl. toggo-plus).
Proxy (st/p1.py): opt_lauf.lade(training, jede 8. s) + baue_k, K-MLP 96/12 Ep.; die 22 Spur-Aufnahmen dieser 6 Sender aus dem Fit,
  Rest bleibt (Sender also nur spurlos gesehen). Fehler nach 9-Zeilen-Glaettung, Schwelle 0.5:
  Seed  mit SigLIP / ohne / Logit-Mittel   (Mittel je Aufn.; Median je Aufn.; Aufn. Mittel besser:schlechter > 0.005)
  2     .057 / .031 / .036 ; .0092/.0088/.0053 ; 5:1
  3     .069 / .070 / .058 ; .0139/.0104/.0101 ; 4:4
  4     .051 / .029 / .040 ; .0070/.0110/.0046 ; 6:1
  Seeds 0/1 (Gesamtfehler): mit .059/.078, ohne .037/.069, FP Sendung .072/.096 -> .043/.085.
  Gegenprobe 22 Spur-Aufn. gut abgedeckter Sender: SigLIP weg kostet stark (.008 -> .020/.074) -> nur Fremdsender mischen.
Lesart: harter Wegfall hilft dem Mittel, schadet aber dem Median (3 von 5 Seeds); das Mittel aus beiden Modi hebt beides.
Vorbehalt: Proxy simuliert 0 Spur-Aufnahmen, real haben nick/comedy 3-5 -> Wirkung im Messgeraet kleiner.
