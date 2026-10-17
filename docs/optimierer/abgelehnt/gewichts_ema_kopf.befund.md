Verlustformen und Architektur geben auf dem Trainingsteil nichts her (Label-Smoothing inert bis schaedlich); einzig ein Gewichts-EMA senkt Fehler und lange Fehlstrecken konsistent, aber klein, weil das 3er-Ensemble die Seed-Streuung schon zum Grossteil wegmittelt.

Daten: saetze()/lade()/baue_k() aus opt_lauf (ohne O25), Trainingsteil 609 Aufn., 506k Zeilen, schritt 4. 2 Falten nach Aufnahme, Platt kreuzweise je Aufnahmehaelfte des Ziel-Folds, 9 Zeilen (36 s) geglaettet, Schwelle 0.5. "lang" = Fehlerzeilen in Strecken >= 8 Zeilen (32 s). Scratchpad ls_diag*.py, read-only.

Label-Smoothing (label-unabhaengige Vertrauensstrafe, Hook gegen expliziten Verlust geprueft, Abw. 2e-8), 1 Kopf:
  Seeds 0-5 Fehler CE 0.0307/0.0318/0.0304/0.0335/0.0309/0.0317 vs eps0.1 0.0303/0.0319/0.0324/0.0325/0.0324/0.0321
  lang CE ~11460 vs eps0.1 ~12000 (+5 %), sicher falsch (p>0.9 falsch) 1.51 % -> 1.66 % (6/6 schlechter)
  eps 0.05/0.2 ebenso; nur Aufnahme-Median der Fehler leicht besser (0.0126 -> 0.0119 in Seed 0/1). -> verworfen, nicht gemessen.
Gewichts-EMA (Abkling 0.99 je Schritt bei 31 Schritten/Epoche = Horizont ~3 Epochen), 1 Kopf, Seeds 0-5:
  Fehler CE Mittel 0.0315 (0.0304-0.0335) vs EMA 0.0307 (0.0304-0.0312); 5/6 besser oder gleich
  lang 11463 -> 11065 (-3.5 %), sicher falsch 1.51 % -> 1.46 % (6/6 besser), |L| p90 10 -> 8.6
Mit 3er-Ensemble wie K, Seeds 10-13:
  Fehler 0.0309/0.0310/0.0311/0.0309 vs 0.0306/0.0301/0.0309/0.0304 (4/4 besser, Mittel -0.0005)
  lang -2.5 % (4/4), Aufnahme-Median der Fehler 0.0119 -> 0.0121 (3/4 minimal schlechter)
Arm nutzt 0.995 je Schritt, weil der volle Trainingsteil 62 statt 31 Schritte je Epoche hat (gleicher Horizont in Epochen).
Fruehere Planer-Messung (focal_verlust_schwere_zeilen.befund): 2 Schichten, LayerNorm, Breite = Rauschen; Focal g1 -5 % Fehler nach Platt.
Klasse: Bandit zog labels (0 Laeufe, aber fuenf Befunde 040-046 und logo_werbebloecke_raus nennen sie ausgeschoepft; Sitzungsvorgabe nur Verlust/Architektur) -> training.
Erwartung klein: Ensemble mittelt den Hauptteil der EMA-Wirkung schon weg; die Gewinnschwelle +0.003 ist unwahrscheinlich.

## Abgelehnt

- **NICHT NEU (Hauptgrund):** Lauf 034-kopf_gewichtsmittel (Arm `arme/kopf_gewichtsmittel.py`) hat exakt diese Hypothese schon gemessen. Gleiche Idee, gleicher Abklingfaktor 0.995 je Adam-Schritt, gleicher Horizont von etwa 3 Epochen bei 62 Schritten je Epoche, gleiche Architektur wie K, gleiches 3er-Ensemble. Der Begründungstext von 034 nennt dieselbe Schwäche, nämlich Seed-Streuung und lange Phantomstrecken.
- **Ergebnis von 034** laut `journal.md` und `urteil.json`: NICHT ERFÜLLT, Median Δ -0.0002, Deltas [-0.0016, -0.0002, 0.002, 0.0041, -0.0025], Teile [F, F, T, T], Aufnahme-Median Δ 0.0, Status `verworfen`.
- Die Begründung des Arms behauptet „Kein Friedhof-Eintrag“ und nennt 034 nirgends. Sie erklärt auch nicht, warum diese Variante etwas anderes misst. Das verlangt die README (Prüfpunkt 3): Eine Variante eines verworfenen Arms ist ein neuer Arm, zählt als weiterer Versuch und braucht einen Grund, der die alte Messung entwertet.
- Der einzige Unterschied zu 034 ist, dass hier die Anlaufkorrektur fehlt. 034 benutzt `min(d,(1+n)/(10+n))`, der neue Arm startet das Mittel dagegen auf den Startgewichten. Das betrifft nur die ersten Schritte, 0.995^744 ≈ 0.02 Restgewicht. Der Arm ist also eine reine Parametervariante von 034, und das Diagnose-Argument (sicher-falsch 6/6 besser, lange Strecken -2.5 bis -3.5 %) ändert daran nichts.
- Der Befund räumt selbst ein, dass die Erwartung klein ist, weil das Ensemble den Hauptteil der EMA-Wirkung wegmittelt, und dass die Gewinnschwelle +0.003 unterschreitet.
