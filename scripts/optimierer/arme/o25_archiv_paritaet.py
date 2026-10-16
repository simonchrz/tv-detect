"""Daten: die 90 O25-Archiv-Aufnahmen aus dem Training nehmen (Produktions-Paritaet).

Die Produktion trainiert seit 24.09. mit --archiv-ausschluss ohne diese 90
Aufnahmen (docs/archiv-ausschluss-o25.json: which=auto, kamen nur ueber den
rec_dir-Defekt 05-03..09-23 ins Archiv, keine eigenen Anker). Die Kontrolle K
traegt 86 davon im Trainingsteil (12 % der Zeilen) — ein benannter Teil der
Abbild-Luecke. O25 wurde damals im alten Tagesserien-Regime (Rauschen ~0.008) gemessen
und als Hygiene ohne Nachweis eingebaut; dieses Messgeraet (sd ~0.001) kann
die Frage zum ersten Mal entscheiden.

Proxy auf dem Trainingsteil (Haelften nach Aufnahme, K-MLP, 3 Seeds je
Richtung, Bewertung auf nicht-maschinellen Aufnahmen der anderen Haelfte):
AUC 0.9674 -> 0.9659 (6/6 schlechter), Sekundenfehler +0.0003, Median je
Aufnahme +0.0005. Erwartet ist also ein kleiner VERLUST; faellt der Arm
deutlich durch, ist die Empfehlung an Simon die Umkehr (Ausschluss im
Nightly aufheben), haelt er, ist O25 als kostenlos belegt.

Die Liste stammt aus einer Produktions-Datei vom 24.09., enthaelt nur
Trainings-/Kalibrier-Aufnahmen (keine aus Entwicklung/Bestaetigung) und
nutzt keine Labels.
"""
import numpy as np

BESCHREIBUNG = ("Daten: 90 O25-Archiv-Aufnahmen (rec_dir-Defekt, which=auto) wie in "
                "der Produktion aus dem Training nehmen")

O25 = frozenset([
    "dvr-anixe-1781518500",
    "dvr-comedy-central-1781199600",
    "dvr-comedy-central-1781286000",
    "dvr-comedy-central-1781545200",
    "dvr-comedy-central-1785153600",
    "dvr-comedy-central-1785844800",
    "dvr-deluxe-music-1781470800",
    "dvr-disney-channel-1780947600",
    "dvr-disney-channel-1780955100",
    "dvr-disney-channel-1781125200",
    "dvr-disney-channel-1781125800",
    "dvr-disney-channel-1781151000",
    "dvr-disney-channel-1781237400",
    "dvr-disney-channel-1781468100",
    "dvr-disney-channel-1781470200",
    "dvr-disney-channel-1781472000",
    "dvr-disney-channel-1781473800",
    "dvr-disney-channel-1781475600",
    "dvr-disney-channel-1787371800",
    "dvr-kabel-eins-1780671469",
    "dvr-kabel-eins-1780675029",
    "dvr-kabel-eins-1780930639",
    "dvr-kabel-eins-1780934189",
    "dvr-kabel-eins-1781017046",
    "dvr-kabel-eins-1781020626",
    "dvr-kabel-eins-1781103440",
    "dvr-kabel-eins-1781107020",
    "dvr-kabel-eins-1781189915",
    "dvr-kabel-eins-1781193495",
    "dvr-kabel-eins-1781279761",
    "dvr-kabel-eins-1781535410",
    "dvr-kabel-eins-1781539018",
    "dvr-kabel-eins-1781621730",
    "dvr-nick-1780945500",
    "dvr-nick-1780947000",
    "dvr-nick-1781118300",
    "dvr-nick-1781204700",
    "dvr-nick-1781376000",
    "dvr-prosieben-1780043100",
    "dvr-prosieben-1780377480",
    "dvr-prosieben-1780666677",
    "dvr-prosieben-1780669800",
    "dvr-prosieben-1780679125",
    "dvr-prosieben-1780811400",
    "dvr-prosieben-1780851927",
    "dvr-prosieben-1780915593",
    "dvr-prosieben-1780922368",
    "dvr-prosieben-1780927353",
    "dvr-prosieben-1780938332",
    "dvr-prosieben-1780987477",
    "dvr-prosieben-1781002211",
    "dvr-prosieben-1781074090",
    "dvr-prosieben-1781088532",
    "dvr-prosieben-1781100189",
    "dvr-prosieben-1781101860",
    "dvr-prosieben-1781111128",
    "dvr-prosieben-1781175203",
    "dvr-prosieben-1781246588",
    "dvr-prosieben-1781283901",
    "dvr-prosieben-1781456612",
    "dvr-prosieben-1781532287",
    "dvr-prosieben-1781543122",
    "dvr-prosieben-1781618739",
    "dvr-prosieben-1781620347",
    "dvr-prosieben-1785085457",
    "dvr-rtl-1779865200",
    "dvr-rtl-1780235100",
    "dvr-rtl-1785003300",
    "dvr-rtlzwei-1780229700",
    "dvr-rtlzwei-1781123700",
    "dvr-rtlzwei-1781338500",
    "dvr-rtlzwei-1781435700",
    "dvr-rtlzwei-1781439600",
    "dvr-rtlzwei-1784974500",
    "dvr-rtlzwei-1785075300",
    "dvr-sat-1-1779890400",
    "dvr-sat-1-1779976800",
    "dvr-sixx-1779867600",
    "dvr-vox-1780934400",
    "dvr-vox-1781024400",
    "dvr-vox-1781107200",
    "dvr-vox-1781110800",
    "dvr-vox-1781193600",
    "dvr-vox-1781197200",
    "dvr-vox-1781280000",
    "dvr-vox-1781283600",
    "dvr-vox-1781453400",
    "dvr-vox-1781542800",
    "dvr-vox-1781629200",
    "dvr-vox-1785082200",
])


def gewichte(ktx):
    rec = ktx["rec"]
    raus = np.array([u in O25 for u in ktx["uuids"]], bool)
    return np.where(raus[rec], 0.0, 1.0).astype(np.float32)
