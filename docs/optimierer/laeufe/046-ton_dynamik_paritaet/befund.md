K rechnet mit dem rohen Ton-Pegel, der deployte Kopf seit O22 mit dessen 30-s-Schwankung (head.audio.json dynamik=true, Spalte ersetzt) -- eine Abbild-Luecke, die das README nicht fuehrt; im K-Proxy senkt die Ersetzung die Sekundenfehler in 4/4 Laeufen leicht.

Klasse labels gezogen (Zug 0.81, weil 0 Laeufe); nicht gebaut: elf fruehere Befunde (032, 040, 041, 044, 045, blockende_logo_nachlauf_kuerzen,
logo_werbebloecke_raus, auto_blockende_logo, rand_einblendung_ziel ...) finden die Trainingslabels sauber, und die verbliebenen Konventionsfaelle
(Rand-Trailer, Gewinnspiel, Nachlauf) stehen im Entwicklungssatz genauso -> ein Label-Arm arbeitet gegen die Messlatte. Ausweichen auf merkmale
(zweithoechster Zug 0.43).

Daten: saetze() ohne O25, 524 Trainingsaufnahmen mit ads, Spalte 1281 aus feature_npy (Archiv nur gelesen). Scratchpad r2dyn/d1.py, p1.py.
* Ton-AUC je Aufnahme (Median, Werbung gegen Sendung): Pegel 0.559 (nuetzlich >0.55 in 56 %), Schwankung 30 s voll 0.596 (61 %),
  Schwankung aus 7 Stuetzstellen alle 4 s (wie der Arm) 0.596 (63 %); gepoolt invers (0.42), wie train-head.audio_dynamik beschreibt.
  Schwaecher als die O22-Messung (0.726 auf 98 Aufn.), aber klar ueber dem Pegel.
* Stuetzstellen-Schaetzer gegen 31-s-Vollfenster: r 0.74, Mittel 0.875x. Der Arm nutzt ihn in Training (schritt 4) UND Eval (schritt 1)
  identisch (synthetisch geprueft: innen bitgleich), damit kein Train/Serve-Bruch im Messgeraet entsteht.
* Proxy: volle K-Spalten (baue_k), HYPER_K mit 1 Kopf, Haelften nach sha1(uuid), 2 Seeds, Schwelle 0.5:
  h0 s0 Fehler 0.0807 -> 0.0803, h0 s1 0.0813 -> 0.0807, h1 s0 0.0629 -> 0.0616, h1 s1 0.0639 -> 0.0636 (4/4 besser);
  FP 16420/16575/11364/11674 -> 16375/16491/11090/11637, FN 4294/4294/4333/4289 -> 4239/4215/4299/4229 (beide Seiten sinken);
  AUC gleich oder +0.0006; Median-Fehler je Aufnahme 2 besser / 2 schlechter (+-0.001).
Erwartung klein: der Kopf hat mit OCR/SigLIP schon viel von dem, was die Schwankung traegt; O22 lag ohne beides bei +0.0024 F1.
Abgrenzung: 036/037 nutzten den 60-s-MEDIAN des Pegels relativ zur Aufnahme als Nachbearbeitung, nie die Streuung als Spalte;
O22 war gegen die Produktion vor OCR/SigLIP gemessen, nie gegen K.
Bei verworfen: Ton-Schwankung traegt neben OCR/SigLIP nichts mehr; Kandidat hiesse nur K-Paritaet (Produktion hat es schon).
