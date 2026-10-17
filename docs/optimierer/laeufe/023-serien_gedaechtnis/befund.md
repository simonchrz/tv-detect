Ein label-bedingtes Gedaechtnis der eigenen Serie (bester Backbone-Cosinus zu Sendungs- bzw. Werbezeilen ANDERER Folgen desselben Titels, leave-one-out) trennt im Trainingsteil allein mit AUC 0.946 und hebt im Proxy die AUC eines K-aehnlichen Kopfs in 4/4 Laeufen; die Klasse labels gab vorher nichts her.

Klasse labels gezogen (Zug 0.61) und geprueft, read-only, Trainingsteil von K (saetze(), ohne O25), nur Summen:
* Auto-Blockenden: letzter Logo-Lauf (> 0.6, geglaettet +-3 s) vor dem Label-Ende >= 12 s bei auto 24.5 % der Bloecke, merged 18.9 %
  (>= 24 s: 16.6 % gegen 10.0 %); nur ~3200 s (0.16 % der Zeilen) -> zu klein, auch wenn der alte Proxy-Einwand von 014 entwertet ist.
* Werbe-Label unter anhaltendem Logo (Lauf >= 12 s, ohne ocr_werbung): auto 12.6 % der Werbe-s, merged 6.0 %, user 7.2 %; aber die
  Spitzentitel sind dieselben wie in reviewten Labels (Beet-Brueder merged 4715 s, Abenteuer Leben 4598 s) -> Logo-Vorlage taugt dort
  nicht, kein Label-Beleg (vgl. 016 logo-blind).
* Kurze Bloecke < 60 s / Sendungsluecken < 180 s: auto 4 / 14, merged 7 / 14 -> nichts. Aufnahmeraender (300 s): Sendung ohne Logo
  7.0 % gegen 5.5-6.7 % in der Mitte -> kein Hebel. confirmed_ad_skips: 175 Punkte, Einzelsekunden -> nichts.
-> Ausweichen auf merkmale (zweithoechster Zug 0.44).

Serien-Gedaechtnis (Backbone zentriert, PCA 64 aus jeder 7. Zeile, max Cosinus je Klasse, eigene Aufnahme ausgeschlossen):
* Abdeckung: 494/527 Trainingsaufnahmen (91.3 % der Zeilen) haben eine Geschwister-Folge; im Entwicklungssatz haben 70/76 Aufnahmen
  >= 1 und 60/76 >= 3 Trainings-Folgen gleichen Titels (nur gezaehlt, keine Labels).
* AUC (s_werbung - s_sendung) 0.946, Logo allein 0.906; Logo > 0.6: 0.865, Logo < 0.2: 0.938, Logo 0.2-0.6: 0.946.
  Je Sender: nick 0.933 (Logo 0.854), rtl 0.933 (0.749), comedy-central 0.892 (0.810), vox 0.971, kabel-eins 0.952.
* Proxy: MLP-96 (8 Ep.) auf Backbone+Logo+Audio mit/ohne die 4 Spalten, Haelften nach sha1(uuid), 2 Seeds x 2 Richtungen,
  Sekundenfehler nach Glaettung +-4 Zeilen:
  AUC 0.9582/0.9543/0.9571/0.9545 -> 0.9648/0.9605/0.9681/0.9677; Fehler 0.0392/0.0511/0.0545/0.0492 -> 0.0275/0.0503/0.0389/0.0471;
  Median je Aufnahme 0.0192/0.0208/0.0241/0.0202 -> 0.0157/0.0205/0.0177/0.0175; besser/schlechter je Aufnahme 84/28, 42/70, 100/13, 54/55.
* Vorbehalt: der Proxy hat kein OCR/SigLIP; eine Richtung (train1) verschlechtert mehr Aufnahmen als sie verbessert.
Skripte: Scratchpad r3/e1-e7.py (Labels), r3/s1-s3.py (Gedaechtnis).
