Whisper-TEXT (Wortrate, Werbewort-Anteil, Wortvielfalt, je absolut und relativ zur Aufnahme, ohne die gelernte prob) senkt im K-gleichen Proxy mit OCR+SigLIP die Sekundenfehler in 4/4 Laeufen um 15-32 % bei hoeherer AUC; Richtung (1) Stoerungs-Robustheit gab nichts her.

Klasse merkmale (Bandit). Richtung (1) zuerst geprueft: die "gestoerte" super-rtl-1781112000 (Schiri-Satz) hat 0 NaN, normale Backbone-Norm/Nachbar-Kosinus, aber Logo 0.00 ueber alle 30 min -> das ist Logo-Ausfall (Familie 009/016/024/025/027, abgegrast), keine Decode-Spur; fruehere Befunde: Trainingsteil ohne gestoerte Strecken. Daher Richtung (2).
Abdeckung Whisper-JSON: training 143/665, kalibrierung 4/40, entwicklung 76/76, bestaetigung 22/22 (Arm greift auf Eval zu 100 %, whisper_da=1).
Ausrichtung: Fenster [t,t+60), Schritt 30; AUC prob gegen Label bei Fenstermitte t+30 = 0.974, bei +-30 s 0.953/0.956.
prob NICHT benutzt: im Proxy AUC 0.983-0.990 = offensichtlich in-sample (Klassifikator auf gelabelten Fenstern trainiert), fuer Entwicklung genauso verdaechtig.
Univariat (14k Fenster, reine Fenster): wortrate AUC 0.315 (Sendung mehr Woerter), werbewort 0.636, wortvielfalt 0.638.
Proxy (Scratchpad w7.py): opt_lauf.lade(training, schritt 8) + baue_k (volle 1350 K-Spalten), standardisiert, K-MLP hidden 96 / 12 Ep., Test = Whisper-Aufnahmen einer sha1-Haelfte, Training = alles andere (Whisper-Spalten 0 ohne JSON, wie im Messgeraet), 2 Seeds:
  Haelfte 0: K err 0.0315 kante 0.1502 ll 0.0996 auc 0.9916 -> +Text 0.0267 / 0.1470 / 0.0881 / 0.9927
  Haelfte 1: K err 0.0742 kante 0.2022 ll 0.2245 auc 0.9832 -> +Text 0.0508 / 0.1922 / 0.1706 / 0.9860
  AUC je Seed: 0.9919/0.9912 -> 0.9929/0.9926; 0.9832/0.9832 -> 0.9864/0.9855.
Ohne Standardisierung (roh, HGB oder MLP) war der Effekt unsichtbar -> Massstab zaehlt, das Messgeraet standardisiert.
Ablation (ohne OCR/SigLIP, 1 Seed): absolute Spalten tragen das meiste (err 0.070->0.051), relative allein etwas weniger (0.056), beide zusammen am besten.
Risiken: Platt sieht nur 4/40 Kalibrier-Aufnahmen mit Whisper (wie SigLIP 6/35); 60-s-Fenster sind grob -> Kanten koennen leiden (Proxy: kante leicht besser); Wortliste fest, nicht gesucht.
Produktionsfrage bei Kandidat: Inferenz braucht Whisper-Transkript je Aufnahme (Kopplung, die O1 loswerden wollte) -- Simon entscheidet.
