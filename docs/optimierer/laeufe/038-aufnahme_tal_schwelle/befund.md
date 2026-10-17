Die beste Schwelle je Aufnahme streut out-of-fold stark (IQR -0.75..+1.25 Logit); das Otsu-Tal des geglaetteten eigenen Logits schaetzt sie ohne Label und hebt die Block-IoU im Proxy um +0.009 (beide Haelften positiv), getragen vom Anheben der Schwelle, wo Sendung nahe 0 liegt.

Klasse daten gezogen (Zug 0.42), aber ausgeschoepft: Vorgabe schliesst Alters-/Aufnahme-/Dubletten-Gewichte aus, Stoerungen sind im
Trainingsteil nicht erkennbar (032, 035, aufnahme_rand_ausblenden), Rand/Film/Herkunft liegen im Friedhof. Daher nachbearbeitung (Zug 0.39).
Daten: saetze()/lade()/baue_k() aus opt_lauf (ohne O25), 527 Aufn., jede 4. s, 506k Zeilen. OOF-Logit eines K-MLP 96/12 Ep.
(Haelften nach sha1(uuid), 2 Seeds gemittelt). Proxy-Dekoder: 32-s-Mittel, Schwelle, Luecken < 88 s schliessen, Bloecke >= 60 s,
Block-IoU wie opt_lauf.block_iou. Skripte Scratchpad q1/oof.py, off.py, v2-v4.py, t2.py (read-only, Summen).
* Headroom: IoU bei Schwelle 0 mittel 0.8935 / median 0.9655; Orakel-Schwelle je Aufnahme 0.9332 / 0.9792; beste globale Schwelle
  nur 0.8975. Sekundenfehler 0.0434, Orakel 0.0224. -> das Problem ist Pegel je Aufnahme, nicht global.
* Schaetzer gegen Orakel-t*: Otsu corr 0.42, Quantil-Mitte 0.38-0.40.
* Otsu ohne Waechter: +0.014, aber auf Pseudo-Aufnahmen ohne Werbung (laengste Sendungsstrecke >= 20 min, n=387)
  Phantombloecke 79 -> 335. Waechter "untere Haeufung < 0 < obere": 79 -> 90, Anteil fehlerfrei 0.88 -> 0.88.
* Mit Waechter, Kappe 2 (= Arm, ueber die Arm-Datei nachgerechnet): IoU 0.8935 -> 0.9021, median 0.9655 -> 0.9669,
  Haelften +0.0127 / +0.0034, Aufnahmen < -0.10: 4, > +0.10: ~24; Sekundenfehler 0.0434 -> 0.0334 (ohne Waechter-Varianten).
* Richtung: Schwelle heben (n=186) +0.023 je Aufn., senken (n=195) +0.002, ~0 (n=146) -0.002. Gewinn = erfundene Werbung,
  die an echte Bloecke anwaechst (kostet IoU ueber die Vereinigung).
* Sender: disney +0.069 (n26), rtl +0.022, prosieben +0.012, nick +0.009, vox -0.002, comedy-central -0.077 (n8!),
  wenig Werbung (< 8 %, n=10) -0.045.
Abgrenzung: 020 aufnahme_pegel_stapel lernte den Aufnahme-Mittelwert als Stapel-Spalte (mit Werbeanteil verwechselt);
005 zentrierte Merkmale. Hier label-frei die Tal-Lage der Ausgabe, unabhaengig vom Werbeanteil.
Risiken: echter HSMM glaettet anders als der Proxy; comedy-central (Entwicklung n4) und Aufnahmen mit wenig Werbung.
Wirkung pruefen: Zahl verschobener Eval-Aufnahmen ist > 0 (Arm verschiebt im Proxy 92 %), nicht wie 035.
