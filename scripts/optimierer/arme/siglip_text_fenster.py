"""SigLIP-Textrichtung im Zeitfenster: sieht die Umgebung semantisch nach
Werbung aus?

Die SigLIP-2-Spur (768-dim, je Sekunde) liegt im selben Raum wie der
Text-Turm desselben Modells. Eine Richtung "Werbung minus Sendung" aus
Textbeschreibungen braucht KEIN Label und keinen Fit: Trainings-, Kalibrier-
und Eval-Zeilen werden exakt gleich behandelt, ohne in-sample-Optimismus
(anders als ein auf dem Trainingsteil gefittetes Fenster-Merkmal).

K sieht SigLIP nur als 64 geweisste Hauptkomponenten je Sekunde; die
Textrichtung liegt nur zu 73-94 % in diesem Unterraum, und vor allem sieht
der Kopf die Sekunde einzeln. Der Fehlerposten "Sendung ohne trennendes
Logo" (Spielfilm, Kindersender) ist ein Pegel-Problem ueber Minuten: dort
entscheidet, wie die UMGEBUNG semantisch aussieht. Teleshopping (015-
Nachbefund) steht in der Werbe-Liste, langsamer Schnitt traegt hier nichts.

Richtung = normierter Mittelwert der Werbe-Texte minus normierter
Mittelwert der Sendungs-Texte (google/siglip2-base-patch16-naflex,
get_text_features, padding max_length 64), offline einmal berechnet und
hier als Konstante abgelegt (float32, base64). Werbe-Texte:
  * a TV commercial advertising a product
  * a packshot of a product with a brand logo
  * a car driving on a road in an advertisement
  * a teleshopping presentation of a product with a price
  * an advertisement for a supermarket with prices
  * a cosmetics advertisement with a woman
  * an advertisement for a medicine or pharmacy product
  * a TV channel trailer announcing a show with date and time
  * a commercial for a mobile phone or internet provider
  * a food or drink commercial
  * an insurance or bank commercial
  * a toy commercial for children
Sendungs-Texte:
  * a scene from a feature film
  * a cartoon for children
  * an animated TV series
  * a TV studio show with a host
  * a reality TV documentary scene
  * people talking in a living room in a TV series
  * a news broadcast
  * a talent show on stage
  * a cooking show
  * a crime series scene with police
  * a documentary about nature or travel
  * people on the street being interviewed

Spalten (4), je Zeile aus z = cos(Spur[sekunde], Richtung):
  z            die Sekunde selbst
  mitte16      Mittel ueber +-16 s
  vor60        Mittel ueber [s-60, s) (Zeilen davor)
  nach60       Mittel ueber (s, s+60] (Zeilen danach)
Fenster in Sekunden, umgerechnet ueber ktx["schritt"] (Training jede 4. s:
+-4 bzw. 15 Zeilen; Kalibrierung/Eval jede s: +-16 bzw. 60 Zeilen).
Sekunden ohne Spur (fehlende Datei oder hinter ihrem Ende) zaehlen im
Fenster nicht; bleibt nichts, gilt das Mittel der Trainingszeilen mit Spur
(neutral, siglip_da sagt dem Kopf ohnehin, ob die Spur da ist).
Die Spur wird nur GELESEN (wie baue_k es fuer K tut). Kein Zufall.
"""
import base64
from pathlib import Path

import numpy as np

BESCHREIBUNG = ("SigLIP-Textrichtung Werbung-minus-Sendung (zero-shot, ohne Fit) je Sekunde "
                "plus Fenster +-16 s, 60 s davor, 60 s danach")

SPUR = Path.home() / ".cache/tvd-siglip2"
MITTE_S = 16
SEITE_S = 60
_RICHTUNG_B64 = (
    "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"
)
RICHTUNG = np.frombuffer(base64.b64decode(_RICHTUNG_B64), dtype="<f4").astype(np.float32)
_STAT = {}


def _spur(u):
    if not (SPUR / f"{u}.npy").is_file() or not (SPUR / f"{u}.json").is_file():
        return None
    return np.load(SPUR / f"{u}.npy", mmap_mode="r")


def _z(u, sek):
    """cos(Spur, Richtung) je Zeile; NaN wo keine Spur."""
    z = np.full(len(sek), np.nan, np.float32)
    E = _spur(u)
    if E is None or len(E) == 0:
        return z
    ok = sek < len(E)
    if ok.any():
        e = np.asarray(E[sek[ok]], np.float32)
        n = np.linalg.norm(e, axis=1)
        z[ok] = np.where(n > 1e-6, (e @ RICHTUNG) / np.maximum(n, 1e-6), np.nan)
    return z


def _fenster(z, lo, hi):
    """NaN-bewusstes Mittel ueber Zeilen [i+lo, i+hi]; NaN wenn leer."""
    n = len(z)
    da = ~np.isnan(z)
    cv = np.r_[0.0, np.cumsum(np.where(da, z, 0.0))]
    cn = np.r_[0, np.cumsum(da)]
    i = np.arange(n)
    a = np.clip(i + lo, 0, n)
    b = np.clip(i + hi + 1, 0, n)
    b = np.maximum(a, b)
    k = cn[b] - cn[a]
    out = np.full(n, np.nan, np.float32)
    m = k > 0
    out[m] = (cv[b] - cv[a])[m] / k[m]
    return out


def spalten(teil, X, ktx):
    schritt = int(ktx["schritt"])
    rec = np.asarray(ktx["rec"])
    sek_alle = ktx.get("sek")
    mitte = max(1, MITTE_S // schritt)
    seite = max(1, SEITE_S // schritt)
    zus = np.full((len(X), 4), np.nan, np.float32)
    for r, u in enumerate(ktx["uuids"]):
        m = np.flatnonzero(rec == r)
        if len(m) == 0:
            continue
        if sek_alle is not None:
            sek = np.asarray(sek_alle)[m].astype(np.int64)
        else:
            sek = np.arange(len(m), dtype=np.int64) * schritt
        z = _z(u, sek)
        zus[m, 0] = z
        zus[m, 1] = _fenster(z, -mitte, mitte)
        zus[m, 2] = _fenster(z, -seite, -1)
        zus[m, 3] = _fenster(z, 1, seite)
    if teil == "training" or "neutral" not in _STAT:
        v = zus[~np.isnan(zus[:, 0]), 0]
        _STAT["neutral"] = float(v.mean()) if len(v) else 0.0
    zus = np.where(np.isnan(zus), _STAT["neutral"], zus).astype(np.float32)
    return np.concatenate([np.asarray(X, np.float32), zus], 1)
