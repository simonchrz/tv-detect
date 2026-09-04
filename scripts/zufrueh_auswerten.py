#!/usr/bin/env python3
"""Wie weit reicht ein zu früher Blockanfang in die Sendung hinein?

Regel aus docs/blockanfaenge-zufrueh-registrierung.md, wortgleich:
Ab dem Bild AM Blockanfang vorwaerts zaehlen, wie viele Bilder in Folge
SENDUNG sind; `zu frueh um X s` = Schritt × diese Zahl.

Das Bild VOR dem Anfang ist die Gegenprobe: ist dort schon Werbung, hat der
Block bereits begonnen — dann ist die Kante nicht "zu frueh", sondern
UNBRAUCHBAR fuer diese Frage. Ohne diese Bedingung wuerde jede Kante mitten
in einem laufenden Werbeblock als sauber durchgehen.
"""
import sys

SCHRITT_S = 4.0
I_VOR = 0        # -4 s
I_START = 1      # 0 s = gelabelter Blockanfang


def urteil(folge, schritt=SCHRITT_S):
    """-> ("zu_frueh", sekunden) | ("gedeckelt", sekunden) | ("ok", 0.0)
          | ("unbrauchbar", 0.0)"""
    if len(folge) < 3:
        return ("unbrauchbar", 0.0)
    if folge[I_VOR] == "W":
        return ("unbrauchbar", 0.0)
    n = 0
    for c in folge[I_START:]:
        if c != "S":
            break
        n += 1
    if n == 0:
        return ("ok", 0.0)
    if I_START + n >= len(folge):
        return ("gedeckelt", schritt * n)
    return ("zu_frueh", schritt * n)


if __name__ == "__main__":
    print("Modul — wird von der Auswertung importiert.", file=sys.stderr)
