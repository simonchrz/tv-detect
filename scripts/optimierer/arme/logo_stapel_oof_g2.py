"""Gestapelter Kopf: ehrliche Bildmeinung gegen Logo/OCR/Audio abwaegen.

Befund (Trainingsteil, haelftig nach Aufnahme getrennt, jede 8. s, K-artiges
MLP auf Haelfte A, Vorhersage auf Haelfte B): der Kopf nutzt das Logo
ausserhalb seiner Trainingsdaten zu wenig. Bei Logo >= 0.8 und p >= 0.8 sind
nur 53 % der Sekunden Werbung, bei p in [0.5, 0.8) nur 23 %; ohne Logo
(< 0.2) bei p in [0.5, 0.8) dagegen 56 %. Die schlimmste Aufnahme (rtl,
Werbeanteil 0.06, Logo-Mittel 0.72) bekommt p 0.81 und 82 % Fehler. Das ist
die Verwechslungszone: der Backbone erinnert sich an Werbe-aehnliches
Material, und das Logo kommt nicht dagegen an. Im Training sieht der Kopf
seine eigene Backbone-Meinung nur in-sample (auswendig gelernt, fast immer
richtig) und lernt deshalb nie, wann er ihr gegen das Logo misstrauen soll.

Eine zweite Stufe auf der OUT-OF-FOLD-Meinung plus den kleinen Spalten
behebt das auf Haelfte B (zweifach kreuzvalidiert nach Aufnahme):
Platt allein logloss 0.1870 / Fehler 0.0633, mit Logo/Audio/OCR/Indikatoren
logloss 0.1725 / Fehler 0.0569 (-10 % Sekundenfehler).

Umsetzung ganz in spalten():
* Stufe 1 = K-artiges MLP (hidden 96, 12 Epochen, Klassengewichte) auf
  allen Kontrollspalten, 4 Falten nach Aufnahme. Training bekommt die
  Meinung des Faltenmodells, das die Aufnahme NICHT gesehen hat;
  Kalibrierung/Eval das Logit-Mittel aller 4 Faltenmodelle.
* Stufe 2 (der normale Kopf des Laeufers) sieht nur: Stufe-1-Logit, logo,
  audio, ocr_hinweis, ocr_werbung, ocr_da, siglip_da. Keine Backbone-
  oder SigLIP-Rohspalten, sonst lernt er wieder in-sample auswendig.

Benutzt nur Merkmale und, im Trainingsteil, dessen Labels (ktx["y"]).
"""
import numpy as np

BESCHREIBUNG = ("Gegenprobe 2 zu 006 (Stufe-1-Seeds 8000, andere Falten) — Stapel: Stufe 1 = K-MLP out-of-fold (4 Falten nach Aufnahme), "
                "Stufe 2 = Kopf auf Stufe-1-Logit + logo/audio/ocr/Indikatoren")

FALTEN = 4
KLEIN = ("logo", "audio", "ocr_hinweis", "ocr_werbung", "ocr_da", "siglip_da")
_ZUSTAND = {}


def _trainiere(Xt, mu, sd, y, idx_train, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = Xt.device
    net = nn.Sequential(nn.Linear(Xt.shape[1], 96), nn.ReLU(), nn.Linear(96, 2)).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    n1 = float((y[idx_train] == 1).sum())
    n0 = float(len(idx_train) - n1)
    kw = torch.tensor([len(idx_train) / (2 * max(n0, 1.0)),
                       len(idx_train) / (2 * max(n1, 1.0))], dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss(weight=kw)
    idx = np.array(idx_train)
    for _ in range(12):
        net.train()
        rng.shuffle(idx)
        for s in range(0, len(idx), 8192):
            j = torch.from_numpy(idx[s:s + 8192]).to(dev)
            opt.zero_grad()
            l = lf(net((Xt[j] - mu) / sd), yt[j])
            l.backward()
            opt.step()
    net.eval()
    return net


def _logit(net, X, mu, sd):
    import torch
    dev = mu.device
    aus = []
    with torch.no_grad():
        for s in range(0, len(X), 65536):
            xb = torch.from_numpy(np.ascontiguousarray(X[s:s + 65536], np.float32)).to(dev)
            o = net((xb - mu) / sd)
            aus.append((o[:, 1] - o[:, 0]).cpu().numpy())
    return np.clip(np.concatenate(aus), -12, 12).astype(np.float32)


def _klein(X, namen):
    return np.asarray(X, np.float32)[:, [namen.index(k) for k in KLEIN]]


def spalten(teil, X, ktx):
    import torch
    namen = ktx["spalten_namen"]
    X = np.asarray(X, np.float32)
    if teil == "training":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        y = np.asarray(ktx["y"], np.int64)
        rec = np.asarray(ktx["rec"])
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        mu_t = torch.from_numpy(mu).to(dev)
        sd_t = torch.from_numpy(sd).to(dev)
        Xt = torch.from_numpy(X).to(dev)
        # Falten nach Aufnahme, fest ueber die Aufnahme-Nummer gestreut.
        falte = (rec.astype(np.int64) * 40503 % 1000003) % FALTEN
        oof = np.zeros(len(X), np.float32)
        netze = []
        for f in range(FALTEN):
            tr = np.flatnonzero(falte != f)
            te = np.flatnonzero(falte == f)
            net = _trainiere(Xt, mu_t, sd_t, y, tr, 8000 + f)
            oof[te] = _logit(net, X[te], mu_t, sd_t)
            netze.append(net)
        del Xt
        _ZUSTAND.update(netze=netze, mu=mu_t, sd=sd_t)
        stufe1 = oof
    else:
        if "netze" not in _ZUSTAND:
            raise RuntimeError("spalten('training') muss vor den anderen Teilen laufen")
        stufe1 = np.mean([_logit(n, X, _ZUSTAND["mu"], _ZUSTAND["sd"])
                          for n in _ZUSTAND["netze"]], 0).astype(np.float32)
    return np.concatenate([stufe1[:, None], _klein(X, namen)], 1)
