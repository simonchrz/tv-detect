Bilddynamik (Backbone-Abstand zu t+-4 s, gemittelt ueber +-30 s) trennt Werbung von Sendung stark und senkt im Proxy den Fehler in den Verwechslungszonen um ein Drittel, auch nach Glaettung der Vorhersage; an Kanten hilft sie nicht.

Daten: Trainingsteil, 240 zufaellige Aufnahmen (Seed 0), 191 mit Merkmalen, 180k Zeilen (jede 4. s), Werbeanteil 0.239.
Merkmal = 1 - Kosinus des zentrierten Backbones zu t-4 s (Training: 1 Zeile).
Einzel-AUC: d_zur 0.667, d_min 0.696, Fenster-Mittel +-30 s 0.868, Still-Anteil 0.351 (inv. 0.649).
Mittel Sendung/Werbung: d_zur 0.65/0.81, Still-Anteil 0.071/0.026.

Proxy: HistGradientBoosting auf Backbone-PCA48 + logo + audio, Aufnahmen haelftig geteilt, beide Richtungen.
Verwechslungszone = 10 NN (PCA48) aus ANDEREN Aufnahmen, >= Haelfte Gegenteil-Label (30k Stichprobe; Anteil 0.13, wie im Memory).
Kante = <= 15 s an Label-Wechsel.
                 Fehler        Zone          Kante
  basis          0.084/0.075   0.312/0.323   0.233/0.238
  +punktweise    0.071/0.065   0.274/0.271   0.225/0.228
  +fenster       0.048/0.047   0.225/0.208   0.235/0.235
  +alle 5        0.049/0.046   0.218/0.209   0.240/0.236
Gegenprobe gegen "nur Glaettung": Vorhersage +-32 s gemittelt —
  basis 0.046/0.045 (Zone 0.170/0.202), +fenster 0.027/0.030 (Zone 0.127/0.138).
  Der Gewinn bleibt nach Glaettung fast voll -> neue Information, kein verkappter Dekoder.
Schritt-Pruefung (12 Aufnahmen, voll vs jede 4. s): Spaltenmittel 0.654/0.654/0.529/0.654/0.082 gegen 0.658/0.658/0.532/0.658/0.083.
Vorbehalt: der Proxy-Kopf ist schwach (kein volles Backbone, kein OCR/SigLIP); der echte Kopf kann einen Teil schon haben.
Risiko: Fenster ueber Kanten verschmiert (Kantenfehler im Proxy unveraendert); 004 (OCR/Logo-Fenster) verlor, super-rtl reagierte dort stark.
Skripte: Scratchpad diag1-3.py (read-only, nur Summen).
