"""O35: Logo-Spalte der nicht korrigierten Archiv-Eintraege neutralisieren.

Befund (Ledger §3ay, 18./19.10.): train-head.py hat die Logo-Vorlage nicht auf
die Bildgroesse umgerechnet. 51 Merkmalsdateien mit lokaler Quelle wurden neu
geschrieben; 65 Eintraege ohne Quelle (Audit-Liste ohne-quelle.json) tragen
weiter den alten Wert. Bei den korrigierten Aufnahmen lag der alte Wert im
Niveau deutlich zu tief: Comedy Central in der Sendung im Mittel 0.30 statt
0.85, Disney 0.43 statt 0.67 (auch wo die Korrelation 0.97+ ist). Im
Trainingsteil sind 41 dieser Eintraege (fast alle Disney-Aufnahmen), im
Kalibrierteil 3, in Entwicklung/Bestaetigung keiner. Die Produktion misst das
Logo immer richtig — der Kopf lernt also auf diesen Sendern eine Skala, die er
live nie sieht.

Arm: fuer genau diese Eintraege wird die Logo-Spalte auf den Platzhalter 0.5
gesetzt ("kein Logo-Signal"), den das Training schon fuer Aufnahmen ohne
Logo-Spalte benutzt. Feste Liste aus dem Audit vom 18.10., kein Label der
bewerteten Aufnahmen, kein Zufall. Wirkt in allen Teilen, praktisch nur in
Training und Kalibrierung.
"""
import numpy as np

BESCHREIBUNG = ("Logo-Spalte der 65 nicht neu berechneten Eintraege (alte, "
                "unskalierte Vorlage, Audit 18.10.) auf Platzhalter 0.5")

STALE = {
    "dvr-comedy-central-1778578800",  # vorlage 1024
    "dvr-comedy-central-1778581800",  # vorlage 1024
    "dvr-comedy-central-1778595300",  # vorlage 1024
    "dvr-comedy-central-1781113200",  # vorlage 1024
    "dvr-comedy-central-1781199600",  # vorlage 1024
    "dvr-comedy-central-1781286000",  # vorlage 1024
    "dvr-comedy-central-1781545200",  # vorlage 1024
    "dvr-comedy-central-1785153600",  # vorlage 1024
    "dvr-comedy-central-1785844800",  # vorlage 1024
    "dvr-comedy-central-1786536000",  # vorlage 1024
    "dvr-disney-channel-1779136800",  # vorlage 1024
    "dvr-disney-channel-1779138600",  # vorlage 1024
    "dvr-disney-channel-1779223500",  # vorlage 1024
    "dvr-disney-channel-1780716900",  # vorlage 1024
    "dvr-disney-channel-1780719000",  # vorlage 1024
    "dvr-disney-channel-1780743300",  # vorlage 1024
    "dvr-disney-channel-1780744500",  # vorlage 1024
    "dvr-disney-channel-1780754400",  # vorlage 1024
    "dvr-disney-channel-1780856100",  # vorlage 1024
    "dvr-disney-channel-1780864200",  # vorlage 1024
    "dvr-disney-channel-1780866300",  # vorlage 1024
    "dvr-disney-channel-1780868100",  # vorlage 1024
    "dvr-disney-channel-1780869900",  # vorlage 1024
    "dvr-disney-channel-1780872000",  # vorlage 1024
    "dvr-disney-channel-1780944000",  # vorlage 1024
    "dvr-disney-channel-1780945800",  # vorlage 1024
    "dvr-disney-channel-1780947600",  # vorlage 1024
    "dvr-disney-channel-1780949700",  # vorlage 1024
    "dvr-disney-channel-1780955100",  # vorlage 1024
    "dvr-disney-channel-1781125200",  # vorlage 1024
    "dvr-disney-channel-1781125800",  # vorlage 1024
    "dvr-disney-channel-1781127300",  # vorlage 1024
    "dvr-disney-channel-1781129100",  # vorlage 1024
    "dvr-disney-channel-1781130900",  # vorlage 1024
    "dvr-disney-channel-1781151000",  # vorlage 1024
    "dvr-disney-channel-1781237400",  # vorlage 1024
    "dvr-disney-channel-1781304900",  # vorlage 1024
    "dvr-disney-channel-1781305140",  # vorlage 1024
    "dvr-disney-channel-1781321700",  # vorlage 1024
    "dvr-disney-channel-1781391900",  # vorlage 1024
    "dvr-disney-channel-1781468100",  # vorlage 1024
    "dvr-disney-channel-1781470200",  # vorlage 1024
    "dvr-disney-channel-1781472000",  # vorlage 1024
    "dvr-disney-channel-1781473800",  # vorlage 1024
    "dvr-disney-channel-1781475600",  # vorlage 1024
    "dvr-disney-channel-1782102900",  # vorlage 1024
    "dvr-disney-channel-1782486900",  # vorlage 1024
    "dvr-disney-channel-1785213300",  # vorlage 1024
    "dvr-disney-channel-1787371800",  # vorlage 1024
    "dvr-disney-channel-1787515200",  # vorlage 1024
    "dvr-disney-channel-1788495000",  # vorlage 1024
    "dvr-disney-channel-1789359000",  # vorlage 1024
    "dvr-disney-channel-1790050200",  # vorlage 1024
    "dvr-disney-channel-1790197500",  # vorlage 1024
    "dvr-disney-channel-1790223000",  # vorlage 1024
    "dvr-disney-channel-1791752700",  # vorlage 1024
    "dvr-disney-channel-1791778200",  # vorlage 1024
    "dvr-kabel-eins-1779980100",  # logoSend 0.10 (Sender 0.99)
    "dvr-mtv-1781663700",  # vorlage 1024
    "dvr-prosieben-1780043100",  # logoSend 0.06 (Sender 0.81)
    "dvr-prosieben-1780062170",  # logoSend 0.09 (Sender 0.81)
    "dvr-prosieben-1780286400",  # logoSend 0.06 (Sender 0.81)
    "dvr-rtlzwei-1780226100",  # logoSend 0.22 (Sender 0.98)
    "dvr-toggo-plus-1781592000",  # vorlage 768
    "dvr-zdf-hd-1781736300",  # vorlage 1280
}


def spalten(teil, X, ktx):
    il = ktx["spalten_namen"].index("logo")
    betroffen = np.array([u in STALE for u in ktx["uuids"]], dtype=bool)
    zeilen = betroffen[np.asarray(ktx["rec"])]
    if zeilen.any():
        X = X.copy()
        X[zeilen, il] = 0.5
    return X
