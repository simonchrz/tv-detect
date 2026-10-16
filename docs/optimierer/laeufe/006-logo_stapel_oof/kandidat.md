# Kandidat 006-logo_stapel_oof: logo_stapel_oof

**Entscheidung:** „übertragen“ oder „nein“. Ein Kandidat ist NICHT
ausgeliefert. Übertragen heißt: in `train-head.py` nachbauen, dann entscheidet
das Nachtgate wie immer (L3).

## Hypothese

In den Verwechslungszonen verlässt sich der Kopf auf seine auswendig gelernte Backbone-Meinung und gibt dem Logo, der OCR und dem Ton zu wenig Gewicht. Im Training sieht er diese Meinung nur auf Daten, die er schon kennt, und dort ist sie fast immer richtig. Deshalb lernt er nie, wann er ihr gegen ein sichtbares Logo misstrauen soll. Ein zweistufiger Kopf behebt das. Stufe 1 ist das K-MLP, außerhalb der eigenen Falte geschätzt (4 Falten nach Aufnahme). Stufe 2 sieht nur das Stufe-1-Logit plus logo, audio, ocr_hinweis, ocr_werbung, ocr_da und siglip_da. Sie lernt also auf ehrlichen Vorhersagen, wie stark die Bildmeinung gegen diese Signale zählt. Damit sollen vor allem Fehlalarme bei sichtbarem Logo sinken (Sendung, die nach Werbung aussieht).

**Begründung:** Fehlerposten: Verwechslungszonen. Der README-Schwachpunkt 3 nennt sie: 13 % der Sekunden tragen 45 % der Fehler. Dazu kommt „NN erfindet“ mit 20 % der Verlust-Sekunden.

Beleg, nur auf dem Trainingsteil gemessen: Ich habe ihn nach Aufnahme in zwei Hälften geteilt (jede 8. Sekunde). Ein K-artiges MLP (2 Köpfe) wurde auf Hälfte A trainiert und hat Hälfte B vorhergesagt (370 Aufnahmen, 140k Zeilen). Ergebnis: Der Fehler hängt nach p noch deutlich vom Logo ab.
- Logo >= 0.8 und p >= 0.8: nur 53 % Werbung.
- Logo >= 0.8 und p in [0.5, 0.8): nur 23 % Werbung.
- Logo < 0.2 und p in [0.5, 0.8): 56 % Werbung.
- Zeilen mit Logo >= 0.5 tragen 57 % aller Fehler, 3437 davon sind Fehlalarme.
- Die schlimmste Aufnahme (rtl, Werbeanteil 0.06, Logo-Mittel 0.72) bekommt p = 0.81 und 82 % Fehler.

Zweite Stufe auf B, zweifach kreuzvalidiert nach Aufnahme:
- Nur Platt: logloss 0.1870, Sekundenfehler 0.0633, AUC 0.9653.
- Mit Logo, Audio, OCR und Indikatoren: logloss 0.1725, Fehler 0.0569, AUC 0.9685. Das sind 10 % weniger Sekundenfehler.
- Logo allein bringt 0.1815.

Warum der Ansatz nicht im Friedhof liegt:
- Er gewichtet nicht um (001/002/O17/O18) und regularisiert nicht (003).
- Er verändert weder die Kapazität (O6) noch die Kontextspalten (O16/004) noch die Zentrierung (005).
- Es ist kein weiches Champion-Ziel (O31): das Ziel bleibt das Menschenlabel. Neu ist nur, dass die Meinung des Kopfs außerhalb der eigenen Falte eine Eingabe wird.
- Der Boundary-Head war ein Kantenmodell, das hier ist eine Stapelung je Sekunde.

Einschränkung: Stufe 1 wird je Lauf einmal mit festen Seeds gefittet. Die 5 Wiederholungen streuen also nur über Stufe 2.

**Beschreibung im Arm:** Stapel: Stufe 1 = K-MLP out-of-fold (4 Falten nach Aufnahme), Stufe 2 = Kopf auf Stufe-1-Logit + logo/audio/ocr/Indikatoren

## Zahlen (Block-IoU über den Produktions-HSMM, gepaart gegen die Kontrolle)

| Satz | Aufnahmen | Kontrolle | Median Δ | Δ je Wiederholung |
|---|---|---|---|---|
| Entwicklung (Messsatz ∩ train) | 76 | 0.9301 | +0.0052 | [0.007, 0.0053, 0.0052, 0.0039, 0.0051] |
| Bestätigung (Messsatz ∩ test) | 22 | 0.9701 | +0.0061 | [0.0061, 0.0064, 0.0065, 0.0058, 0.0059] |

Rauschen zum Vergleich: eine Wiederholung schwankt um ±0.001. Abbild-Lücke
der Kontrolle zur Produktion: −0.009 (16.10.).

**Größte Gewinne (Entwicklung):**

| Aufnahme | Sendung | Δ |
|---|---|---|
| dvr-nick-1782930900 | South Park (nick) | +0.247 |
| dvr-comedy-central-1778764500 | South Park (comedy-central) | +0.177 |
| dvr-comedy-central-1778837700 | South Park (comedy-central) | +0.076 |

**Verluste über 0.10 (Entwicklung):**

| Aufnahme | Sendung | Δ |
|---|---|---|
| – | – | – |

## Wie belastbar

Bis zu diesem Kandidaten wurde der Entwicklungssatz 6-mal und der
Bestätigungssatz 1-mal befragt. Je höher die erste Zahl, desto eher
ist ein knapper Entwicklungsgewinn Glück; die Bestätigung auf ungesehenen
Aufnahmen ist deshalb das eigentliche Kriterium.

## Was die Übertragung kostet

* **spalten:** NEUE EINGABESPALTEN: Training UND Go-Inferenz (nn.go), Header-Bump, Paritaets-Fixture — L5, braucht dein ausdrueckliches OK.

## Code des Arms (`scripts/optimierer/arme/logo_stapel_oof.py`, Abdruck 742e19e6e93a)

```python
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

BESCHREIBUNG = ("Stapel: Stufe 1 = K-MLP out-of-fold (4 Falten nach Aufnahme), "
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
        falte = (rec.astype(np.int64) * 2654435761 % 1000003) % FALTEN
        oof = np.zeros(len(X), np.float32)
        netze = []
        for f in range(FALTEN):
            tr = np.flatnonzero(falte != f)
            te = np.flatnonzero(falte == f)
            net = _trainiere(Xt, mu_t, sd_t, y, tr, 7000 + f)
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
```

## Gegenprobe (16.10., Simon) — NICHT übertragen

Zwei Läufe mit anderen Stufe-1-Seeds und anderer Falten-Zuteilung (007, 008):
Median Δ je **+0.0027** (5/5 positiv), knapp unter der Schwelle. Der Effekt
ist echt, aber halb so groß wie in 006 (dort Glück der ersten Seeds) und
getragen von drei South-Park-Aufnahmen (+0.07 bis +0.24). Der Median über
die Aufnahmen SINKT dabei (Arm 0.967–0.972 gegen K 0.972–0.975): viele
kleine Verluste gegen wenige große Gewinne; in 008 verliert Super RTL
1781581800 −0.159. Das Nachtgate zählt Aufnahmen besser/schlechter und würde
das eher ablehnen. Die L5-Übertragung lohnt so nicht.
