#!/usr/bin/env python3
"""War an diesem Label ein Mensch? — die EINE Definition.

Diese Frage wird an drei Stellen gestellt (`train-head.py` für die
Gewichtung, `massstab-audit.py` für den Maßstab, `golden_v3_vorschlag.py`
für die Kandidatenwahl), und sie war dreimal getrennt beantwortet. Am
2026-09-06 hat eine Mutationsprobe an anderer Stelle gezeigt, wohin das
führt: eine Regel in zwei Kopien ist zwei Regeln, sobald eine angefasst
wird. Deshalb hier, einmal.

WORAN MAN ES ERKENNT
--------------------
NICHT an `which` aus `train-head.py`. Das entsteht aus der blossen
EXISTENZ von `ads_user.json` — und `autoConfirmApply` in
`tv-recorder/autoconfirm.go` legt genau so eine Datei an, mit der
Detektorausgabe unverändert darin. Gemessen an 234 lebenden Aufnahmen mit
nicht-leerem `ads_user.json`: 139 menschlich, 78 maschinell, 17
agentengeschrieben — alle drei Gruppen tragen `which="merged"`.

Erkennbar ist es an den MARKERN in der Datei:

  * `auto_confirmed_at`              — auto-confirm hat bestätigt
  * `auto_confirmed_via_fingerprint` — Fingerprint-Bestätigung; schreibt die
    Modellausgabe MIT `reviewed_at`, aber OHNE `auto_confirmed_at`, sieht
    also menschlich aus (Memory fingerprint_bestaetigung_ist_kein_mensch)
  * `reviewed_by` in NICHT_MENSCH   — ein Werkzeug, kein Mensch

`golden-audit` steht bewusst NICHT in NICHT_MENSCH: es korrigiert
Golden-Labels nach menschlicher Entscheidung.

⚠️ Drei Werte, nicht zwei. `None` heisst **nicht entscheidbar** (keine
lesbare Quelle mehr), nicht „kein Mensch". Wer die beiden zusammenwirft,
degradiert 294 tote train-Aufnahmen auf eine Vermutung.
"""

# Wer einen neuen maschinellen Schreiber ergaenzt, ergaenzt ihn HIER.
NICHT_MENSCH = {
    "agent-review.py",
    "claude-code",
    "zurueckgenommen",
    "folgen-vergleich.py",
}


def mensch_aus_markern(user_raw):
    """True / False / None (nicht entscheidbar) aus einem ads_user.json-Dict."""
    if not isinstance(user_raw, dict):
        return None
    if user_raw.get("auto_confirmed_at"):
        return False
    if user_raw.get("auto_confirmed_via_fingerprint"):
        return False
    if user_raw.get("reviewed_by") in NICHT_MENSCH:
        return False
    return True
