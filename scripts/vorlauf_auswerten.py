#!/usr/bin/env python3
"""Liegt vor einem gelabelten Blockanfang schon Werbung?

Regel aus docs/blockanfaenge-vorlauf-registrierung.md, wortgleich:
ZU SPAET, wenn (1) beide Kontrollbilder NACH dem Anfang WERBUNG sind UND
(2) die zwei unmittelbar davor auch. "Zu spaet um X s" = Laenge der
ununterbrochenen WERBUNG-Kette rueckwaerts vom Anfang.

⚠️ Die Kontrollbilder sind kein Beiwerk. Haelt der Agent auch den Block
SELBST fuer Sendung, ist sein Urteil fuer diese Kante unbrauchbar — dann
zaehlt sie als "unklar", nicht als Beleg. Ohne diese Bedingung wuerde ein
Agent, der durchgehend "SENDUNG" sagt, als Beleg gegen das Label gelten.
"""
import sys


def urteil(folge, i_start):
    """folge: 'W'/'S' je Bild, zeitlich geordnet. i_start: Index des Bildes
    AM Blockanfang (das erste, das laut Label zur Werbung gehoert).
    -> ("zu_spaet", n_bilder) | ("unklar", 0) | ("ok", 0)"""
    nach = folge[i_start + 1:i_start + 3]
    if len(nach) < 2 or any(c != "W" for c in nach):
        return ("unklar", 0)
    if folge[i_start] != "W":
        return ("unklar", 0)
    kette = 0
    for i in range(i_start - 1, -1, -1):
        if folge[i] != "W":
            break
        kette += 1
    if kette < 2:
        return ("ok", 0)
    return ("zu_spaet", kette)


if __name__ == "__main__":
    print("Modul — wird von der Auswertung importiert.", file=sys.stderr)
