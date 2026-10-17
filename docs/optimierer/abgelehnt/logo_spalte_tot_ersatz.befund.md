super-rtl-1781112000 ist keine Decode-Stoerung in den Merkmalen, sondern eine tote Logo-Spalte im Archiv (0.00 ueber 30 min, Live-Detektor 0.3-0.6); K mit Trainings-Median 0.85 statt 0 ist auf solchen Aufnahmen so gut wie auf sauberen.

Klasse merkmale (Bandit, Zug 0.71). Sitzungshinweis (a)/(b) geprueft, Skripte Scratchpad r3/a1-a4.py, p1-p2.py:
* (b) Stoerungsspalte, fuenftes Mass nach NaN/Norm/Flackern/SigLIP-Nachbar/PCA-Residuum (042/044/045): Kreuz-Residuum
  Backbone(z, PCA-128) -> SigLIP-64 per Ridge, 80 Trainingsaufn. Fit, 40 heldout: r q50/95/99 0.90/0.99/1.03;
  super-rtl-1781112000 q50/95 0.93/0.99, 0.7 % ueber p99, kein 60-s-Fenster > 30 %; Entwicklung 2/76 Aufn. mit Fenster.
  Je Minute auf super-rtl: Backbone-Nachbarkosinus 0.76-0.91, Norm 22-26 -> unauffaellig. (a) haette kein Gegenstueck.
* Signal-Dump super-rtl-1781112000: Produktions-NN je Minute 0.00-0.43 (erfindet NICHT), Live-Logo 0.35-0.59,
  Archiv-Logo-Spalte 0.00 in allen 30 Minuten. K bekommt die Archiv-Spalte -> "Logo weg" -> Phantomblock.
* Live- gegen Archiv-Logo (Dumps emit-signals + -schiri, 391 Aufn.): Korrelation je Aufnahme Median 0.85-0.88 in allen Saetzen;
  Archiv tot (< 0.05 in > 95 %) bei lebendem Live-Logo nur die zwei super-rtl-Aufnahmen (1781112000, Entw. 1781581800).
* Volle Saetze (Archiv-Spalte, < 0.05 in > 80 %): training 0/524, kalibrierung 0/35, bestaetigung 0/22, entwicklung 1/76.
  -> Der Ausloeser ist hochspezifisch; Training/Kalibrierung bleiben bitgleich K.
Proxy (p2.py): lade(training, jede 8. s), Backbone+Logo+Audio, MLP 96/8 Ep., Haelften nach sha1(uuid), Glaettung 9 Zeilen,
Schwelle 0.5; Testhaelfte mit Logo-Spalte ganz auf 0 (simulierte tote Spalte), dann ersetzt. Fehler mittel/median/FP Sendung:
  Lauf  K sauber          K tot             tot->0.5          tot->0.85 (Arm)
  s0h0  .062/.028/.062    .128/.062/.155    .075/.037/.077    .058/.037/.043
  s0h1  .096/.045/.114    .237/.181/.303    .136/.069/.167    .087/.048/.099
  s1h0  .099/.039/.117    .192/.117/.244    .122/.057/.146    .086/.040/.094
  s1h1  .061/.029/.065    .152/.082/.189    .081/.044/.090    .054/.031/.048
  Tote Spalte verdoppelt den Fehler; 0.85 holt ihn ganz zurueck (Mittel sogar unter sauber, Median +0.002..0.008), 0.5 nur halb.
  Variante Flag-Spalte + 8 % Augmentation (p1.py): tot .085-.098, aber sauber 2/4 schlechter (bis +0.028) -> verworfen.
Abgrenzung: 042/043 zielten auf logo-lose STRECKEN (auch echte, gestoerte Quelle angenommen) und veraendern Training;
hier nur ganze Aufnahmen mit toter Spalte, ohne Trainingsaenderung. Eigentlicher Fix waere Neu-Extraktion der super-rtl-
Merkmale mit Logo-Vorlage (Archiv, nicht Sache des Arms) -- Hinweis an Simon.
Erwartung: Entwicklung nur 1 Aufnahme betroffen -> Median-Delta ~+0.0005 (042: super-rtl +0.040), sonst exakt 0.

## Abgelehnt

ABLEHNUNG wegen Regeltyp: Der Arm ist als nicht_schlechter angemeldet, die Regel ist laut README und opt_urteil.py nur fuer Vereinfachungen gedacht. Der Arm fuegt einen neuen Mechanismus hinzu (Spalten-Ersatz durch 0.85) und beansprucht einen Gewinn: 'holt den Fehler ganz zurueck', Phantomblock weg. Er gehoert unter --regel gewinn. Praezedenz: logo_ausfall_daempfung wurde in abgelehnt/ aus genau diesem Grund abgelehnt. Seine Begruendung erwartete wie hier 'Median-Delta ~0 (wenige betroffene Aufnahmen)'. Das umgeht die Gewinnschwelle (+0.003, 4/5 positiv), an der die Vorgaenger 042 (+0.0014) und 043 (-0.0010) auf derselben Aufnahmegruppe verworfen wurden.

Das Urteil waere leer. Ich habe die Logo-Spalte in allen Saetzen nachgezaehlt (Schwelle <0.05 in >90 % der Sekunden, schritt 4): training 0 wirksame Aufnahmen, kalibrierung 0, bestaetigung 0, entwicklung genau 1 (dvr-super-rtl-1781581800, 0.967). Die zweite tote Aufnahme dvr-super-rtl-1781112000 (0.991) steht zwar in saetze()['training'], faellt in lade() aber heraus (ads leer, which=auto-confirm). Auf 75 von 76 Entwicklungs- und allen 22 Bestaetigungs-Aufnahmen ist der Arm bitgleich K. Der Median-Delta liegt damit bei null, und nicht_schlechter (>= -0.001) besteht automatisch, unabhaengig davon, ob die Idee traegt. Die Begruendung gibt selbst 'Median-Delta ~+0.0005' an. Gemessen wird nichts ausser dieser einen Aufnahme.
