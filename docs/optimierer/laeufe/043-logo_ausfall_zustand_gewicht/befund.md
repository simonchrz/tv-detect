Echte logo-lose Strecken >= 10 min (gestoerte/logo-lose Quelle) sind im Fit 3x seltener als in Entwicklung/Bestaetigung; sie labelunabhaengig x3-x6 zu gewichten senkt im Proxy die Phantom-Werbung dort deutlich (Haelfte 1: 0.60->0.19, 0.35->0.12) und den Gesamtfehler in 4/4 Laeufen, ohne logo-lose Werbung zu verpassen.

Klasse daten (Bandit, Zug 0.57); Richtung Robustheit gegen gestoerte Quellen (Variante 2 nach 042, hier ohne Spalte und ohne Augmentation: nur echte Zeilen nach Zustand umgewichtet).
Vorkommen (Archiv-Features, 1-s-Raster, read-only, Scratchpad r2/d1-d3.py):
* Andere Stoerspuren gibt es nicht: Standbild-Laeufe >= 20 s (Kosinus > 0.9995) 0 in allen Saetzen; Ton flach >= 60 s: training 3 (11.6k s), entwicklung 1 (190 s), bestaetigung 0.
* Logo aus (31-s-Median < 0.15), Anteil der Sekunden je Laenge, training/entwicklung/bestaetigung:
  1-3 min .088/.099/.085 (Werbeanteil train 0.97); 3-7 min .064/.070/.072 (0.94); 7-10 min .005/.006/.015 (0.85);
  10-20 min .0028/.0096/.0177 (0.24); >= 20 min .0020/.0053/0 (0.15). -> >= 10 min: 0.48 % gegen 1.5 % / 1.8 %.
* Die 44 leeren auto-confirm-Aufnahmen (trainiert die Produktion, K nicht) tragen 3835 s solcher Strecken in 3 Aufnahmen
  (darunter dvr-super-rtl-1781112000, ganz ohne Logo) -> Abbild-Luecke genau in diesem Zustand.
Proxy (r2/p1.py): opt_lauf.lade(training, jede 4. s), Backbone+Logo+Audio, K-MLP 96/10 Ep., Haelften nach sha1(uuid), 2 Seeds,
Glaettung 9 Zeilen, Schwelle 0.5. err / Median je Aufn. / FP in Sendung der langen Strecken / Miss logo-lose Werbung (< 7 min):
  s0h0 K .0416/.0161/.734/.001  x3 .0411/.0154/.697/.001  x6 .0410/.0156/.688/.001
  s0h1 K .0350/.0146/.601/.003  x3 .0339/.0149/.271/.003  x6 .0345/.0147/.186/.003
  s1h0 K .0439/.0164/.760/.002  x3 .0432/.0159/.721/.002  x6 .0429/.0156/.707/.002
  s1h1 K .0389/.0152/.351/.002  x3 .0370/.0145/.144/.003  x6 .0351/.0139/.117/.003
  Grenze 7 min statt 10 (x3): uneinheitlich (s0h0 Median schlechter) -> 10 min, Faktor 4 dazwischen.
  Simulierter 10-30-min-Logo-Ausfall (042-Stil) bewegt sich kaum (FPsim -0.003 bis -0.024): der Arm lehrt Sendung in ECHTEN
  logo-losen Strecken (Backbone-Nachbarschaft), nicht ein allgemeines Logo-Gatter.
Haelfte 0 enthaelt eine 1150-Zeilen-Strecke, die K und Arm fast ganz als Werbung sehen (Spielfilm/Kinder-Stil?) -> Wirkung haengt am Bildstil.
Abgrenzung: 016 wertete ganze Aufnahmen mit Logo-AUC < 0.75 auf (beide Klassen, auch logo-lose Werbung) -> hier nur der Zustand
"Logo >= 10 min aus", der fast nie Werbung ist; 025/042 veraendern Merkmale, hier nicht; leer_aufnahmen_ersatz wertete Sendung
aehnlicher Titel auf, ohne Zustandsbezug.
Erwartung: klein im Median (nur 3 Entwicklungsaufnahmen betroffen), Wirkung ueber nn_erfindet und den Schiedsrichter-Satz.
