Eine label-freie SigLIP-Textrichtung (Werbung minus Sendung, zero-shot aus dem Text-Turm) trennt im Zeitfenster fast so gut wie der ganze Kopf (AUC +-60 s 0.968) und senkt im Proxy die Sekundenfehler in 4 von 4 Laeufen um 16-40 %, in logo-blinden Aufnahmen um ein Drittel.

Klasse labels gezogen und ehrlich geprueft, ohne Arm (dritter Anlauf nach 013/019):
* Werbebloecke mit Bild-Wiederkehr (018-max Median > 0.5) fern vom Rand: merged/user-logo 0 von 814, auto 12 (Let's Dance, Bella Italia, Beet-Brueder). Logo-Bloecke (Mittel >= 0.7): auto 17/171, merged 10/902.
* Proxy (MLP 96, 8 Ep., Backbone+Logo+Audio, Haelften nach sha1(uuid), Eval nur Mensch-Aufnahmen): die 922 Zeilen (auto, Logo-Block) auf Sendung umlabeln -> Fehler 0.0313/0.0430/0.0293/0.0442 -> 0.0319/0.0421/0.0298/0.0434, Median je Aufn. schlechter. Rauschen wie 013.
-> Ausweichen auf merkmale (zweithoechster Zug 0.38).

Zero-shot-Richtung: 12 Werbe-Texte (Spot, Packshot, Teleshopping mit Preis, Trailer mit Sendetermin, ...) und 12 Sendungs-Texte (Spielfilm, Zeichentrick, Studio, Reality, ...), siglip2-base-patch16-naflex lokal; Richtung = norm. Mittel Werbung - norm. Mittel Sendung. Spur = get_image_features (gleicher Raum).
Trainingsteil mit Spur: 163 Aufn., 158k Zeilen (jede 4. s); logo-blind = AUC(Logo-Abwesenheit) < 0.75: 11 Aufn.
* AUC je Sekunde 0.873 (logo-blind 0.810), +-16 s 0.951 (0.864), +-60 s 0.968 (0.871); Median der Aufnahme-AUC +-60 s 0.993.
* Einzelne Texte liegen zu 73-94 % (R2) im PCA-64-Unterraum von K; die Sekunde allein bringt im Proxy wenig (Fehler -3 %, AUC flach).
* Proxy (MLP 96, 10 Ep., Backbone+Logo+Audio+PCA64, Glaettung 9 Zeilen, Haelften, je 2 Seeds), Fehler / Median je Aufn. / logo-blind:
  basis  0.0634/0.0080/0.397  0.0645/0.0076/0.403  0.0126/0.0070/0.086  0.0118/0.0059/0.073
  +4 Sp. 0.0409/0.0072/0.268  0.0386/0.0067/0.250  0.0100/0.0056/0.046  0.0099/0.0054/0.045
  Kanten-Fehler (+-4 Zeilen um wahre Kante) +2..+7 %, der Gewinn liegt im Inneren (NN erfindet/verpasst).
* Gegenprobe: dasselbe Fenster ueber eine auf dem Trainingsteil GEFITTETE lineare Richtung (Logistik auf PCA64): Haelfte A blind 0.40 (kein Gewinn), Haelfte B stark -> in-sample-Optimismus; die Text-Richtung ist in beiden Haelften robust.
Vorbehalte: in K tragen nur 27 % der Trainingszeilen die Spur (Rest neutral), Kalibrierung 6/34; der Proxy hat keinen HSMM, der einen Teil des Fenster-Gewinns schon abholt (vgl. 006/020). Kanten koennten leicht leiden.
Skripte: Scratchpad d1-d5.py (labels), t1/t2.py (Text-Turm, siglip-exp-venv), z1-z6.py (read-only, nur Summen).
