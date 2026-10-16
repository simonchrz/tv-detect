"""Trainings-Regime an den Erkennungs-Regime angleichen: Aufnahmen mit OCR- UND
SigLIP-Spur hoeher gewichten.

Befund (Trainingsteil): nur 161 von 609 Aufnahmen (27 % der Zeilen) tragen
beide Spuren (ocr_da und siglip_da irgendwo > 0). Bei der Erkennung haben
sie heute ALLE (98 von 98 Bewertungsaufnahmen). Der Kopf lernt die
Text-/SigLIP-Spalten also aus einem Viertel seiner Daten und verbringt drei
Viertel seines Verlusts im Regime "keine Spur", das im Betrieb nicht mehr
vorkommt. Gewicht 4 je Zeile einer Aufnahme mit beiden Spuren hebt deren
Verlustanteil auf rund 60 %, ohne die uebrigen Aufnahmen (Backbone-Vielfalt)
wegzuwerfen.

Proxy (2 Haelften nach Aufnahme, K-artiges MLP, 3 Seeds je Richtung,
Fehler auf den voll bespurten Aufnahmen der anderen Haelfte):
Faktor 1 0.0859 / 0.0313, Faktor 4 0.0827 / 0.0295 (-4 % / -6 %);
nur bespurte Aufnahmen (Rest weg) schadet dem Aufnahme-Median deutlich.

Nur Merkmale (Indikator-Spalten) und Aufnahme-Zuordnung; keine Labels.
"""
import numpy as np

BESCHREIBUNG = ("Daten: Aufnahmen mit OCR- und SigLIP-Spur (wie jede Erkennung) "
                "x4 gewichtet, damit der Kopf im Betriebs-Regime lernt")

FAKTOR = 4.0
_ZUSTAND = {}


def spalten(teil, X, ktx):
    if teil == "training":
        namen = ktx["spalten_namen"]
        io, isg = namen.index("ocr_da"), namen.index("siglip_da")
        rec = ktx["rec"]
        n_rec = int(rec.max()) + 1
        ocr = np.zeros(n_rec, bool)
        sig = np.zeros(n_rec, bool)
        np.logical_or.at(ocr, rec, X[:, io] > 0)
        np.logical_or.at(sig, rec, X[:, isg] > 0)
        _ZUSTAND["voll"] = ocr & sig
    return X


def gewichte(ktx):
    rec = ktx["rec"]
    voll = _ZUSTAND.get("voll")
    if voll is None or len(voll) <= int(rec.max()):
        return np.ones(len(rec), np.float32)
    return np.where(voll[rec], FAKTOR, 1.0).astype(np.float32)
