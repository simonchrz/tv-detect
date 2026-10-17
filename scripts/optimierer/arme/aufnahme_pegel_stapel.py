"""Aufnahme-Pegel im Stapel: die ehrliche Bildmeinung der GANZEN Aufnahme als
Kontext, additiv und linear in der zweiten Stufe.

Befund 016: in den logo-blinden Aufnahmen (Spielfilme, Beet-Brueder,
Kindersender) ORDNET das NN die Sekunden gut (Median-AUC 0.96), liegt aber als
Ganzes auf der Werbeseite -- ein Pegel-Problem, kein Rang-Problem. Gewichten
(016, +0.0029) und Logo-Verteilungsspalten im Hauptkopf (009, +0.0012) holen
das kaum, weil der Hauptkopf in-sample trainiert: dort ist jede Aufnahme schon
auswendig gelernt, Aufnahme-Kontext traegt keinen Gradienten.

Hier ist die Aufnahme-Information neu und ehrlich:
* Stufe 1 = K-artiges MLP, 4 Falten nach Aufnahme x 2 feste Seeds
  (7000+f, 7100+f; kein Zufall aus dem Wiederholungs-Seed, Stufe 1 ist je
  Lauf fest wie in 006). Training bekommt die out-of-fold-Meinung,
  Kalibrierung/Eval das Logit-Mittel aller 8 Modelle.
* Je Aufnahme, nur aus Stufe-1-Logit z und Logo derselben Aufnahme (ohne
  Label, verteilungsbasiert, also bei schritt 4 und 1 gleich):
  NN-Werbeanteil mean(sigmoid z), Median z, Logo-Mittel, corr(z, Logo),
  NN-Werbeanteil bei sichtbarem Logo. corr(z, Logo) erkennt logo-blinde
  Aufnahmen ohne Label mit AUC 0.93.
* Stufe 2 = kleines MLP auf [z, logo, audio, ocr_*, siglip_da] PLUS ein
  LINEARER Term auf die 5 Aufnahme-Spalten und ihre Produkte mit z und
  (logo - 0.5). Linear, damit die Stufe 2 die 609 Aufnahme-Konstanten nicht
  als Aufnahme-Kennung auswendig lernt: sie kann nur Pegel und Logo-Vertrauen
  je Aufnahme verschieben.

Proxy auf dem Trainingsteil (haelftig nach Aufnahme, OOF, Glaettung 9 Zeilen),
3 Seeds: Sekundenfehler 0.0295 (Stufe 1) / 0.0306 (Stapel 006) ->
0.0274/0.0265/0.0270; FP 0.0117 -> ~0.0073; logo-blind 0.136 -> ~0.112;
Aufnahme-Median unveraendert (0.0122 -> 0.0121; 006: 0.0134).
"""
import numpy as np

BESCHREIBUNG = ("Stapel mit Aufnahme-Pegel: Stufe 1 K-MLP out-of-fold (4 Falten x 2 Seeds), "
                "Stufe 2 = MLP auf Stufe-1-Logit + kleine Spalten + linearer Term auf "
                "5 Aufnahme-Kontextspalten (NN-Anteil, Median, Logo, corr) und Produkte")

FALTEN = 4
SEEDS1 = (7000, 7100)
KLEIN = ("logo", "audio", "ocr_hinweis", "ocr_werbung", "ocr_da", "siglip_da")
N_MLP = 1 + len(KLEIN)          # Spalten fuer das MLP der Stufe 2
N_KTX = 5                       # Aufnahme-Kontextspalten
N_LIN = 3 * N_KTX               # Kontext, Kontext*z, Kontext*(logo-0.5)
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


def _kontext(z, logo, rec, schritt):
    """5 Aufnahme-Konstanten je Zeile, nur aus z und Logo derselben Aufnahme."""
    rec = np.asarray(rec)
    sig = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    K = np.zeros((len(z), N_KTX), np.float32)
    for i in np.unique(rec):
        m = np.flatnonzero(rec == i)
        zi, li, si = z[m], logo[m], sig[m]
        c = 0.0
        if len(m) > 2 and zi.std() > 1e-6 and li.std() > 1e-6:
            c = float(np.corrcoef(zi, li)[0, 1])
        lp = li > 0.5
        anteil_lp = float(si[lp].mean()) if lp.sum() * schritt > 20 else float(si.mean())
        K[m] = [si.mean(), np.median(zi), li.mean(), c, anteil_lp]
    return K


def _lineare_spalten(z, logo, K):
    kz = (K - _ZUSTAND["k_mu"]) / _ZUSTAND["k_sd"]
    zs = ((z - _ZUSTAND["z_mu"]) / _ZUSTAND["z_sd"])[:, None]
    return np.concatenate([kz, kz * zs, kz * (logo[:, None] - 0.5)], 1).astype(np.float32)


def spalten(teil, X, ktx):
    import torch
    namen = ktx["spalten_namen"]
    X = np.asarray(X, np.float32)
    rec = np.asarray(ktx["rec"])
    klein = X[:, [namen.index(k) for k in KLEIN]]
    logo = X[:, namen.index("logo")]
    if teil == "training":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        y = np.asarray(ktx["y"], np.int64)
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        mu_t = torch.from_numpy(mu).to(dev)
        sd_t = torch.from_numpy(sd).to(dev)
        Xt = torch.from_numpy(X).to(dev)
        falte = (rec.astype(np.int64) * 2654435761 % 1000003) % FALTEN
        z = np.zeros(len(X), np.float32)
        netze = []
        for f in range(FALTEN):
            tr = np.flatnonzero(falte != f)
            te = np.flatnonzero(falte == f)
            if len(te) == 0 or len(tr) == 0:
                continue
            acc = np.zeros(len(te), np.float32)
            for s0 in SEEDS1:
                net = _trainiere(Xt, mu_t, sd_t, y, tr, s0 + f)
                acc += _logit(net, X[te], mu_t, sd_t)
                netze.append(net)
            z[te] = acc / len(SEEDS1)
        del Xt
        K = _kontext(z, logo, rec, ktx["schritt"])
        # Aufnahme-Ebene: Mittel/Streuung ueber Aufnahmen, nicht ueber Zeilen.
        _, erste = np.unique(rec, return_index=True)
        k_sd = K[erste].std(0, keepdims=True)
        k_sd[k_sd < 1e-6] = 1.0
        _ZUSTAND.update(netze=netze, mu=mu_t, sd=sd_t,
                        k_mu=K[erste].mean(0, keepdims=True), k_sd=k_sd,
                        z_mu=float(z.mean()), z_sd=float(max(z.std(), 1e-6)))
    else:
        if "netze" not in _ZUSTAND:
            raise RuntimeError("spalten('training') muss vor den anderen Teilen laufen")
        z = np.mean([_logit(n, X, _ZUSTAND["mu"], _ZUSTAND["sd"])
                     for n in _ZUSTAND["netze"]], 0).astype(np.float32)
        K = _kontext(z, logo, rec, ktx["schritt"])
    return np.concatenate([z[:, None], klein, _lineare_spalten(z, logo, K)], 1)


def modell(n_in, hyper):
    import torch.nn as nn

    class PegelKopf(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(N_MLP, hyper["hidden"]), nn.ReLU(),
                                     nn.Linear(hyper["hidden"], 2))
            self.lin = nn.Linear(n_in - N_MLP, 2)

        def forward(self, x):
            return self.mlp(x[:, :N_MLP]) + self.lin(x[:, N_MLP:])

    assert n_in == N_MLP + N_LIN, n_in
    return PegelKopf()
