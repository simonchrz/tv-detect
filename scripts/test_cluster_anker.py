#!/usr/bin/env python3
"""O24: `--cluster-anker aus` muss die Anker aus JEDEM Fit nehmen.

Gefunden 2026-09-22. Die Anker kommen je Fit verschieden an, weil
`y_train_parts` die Labels per bool-Maske KOPIERT, bevor die Anker-Schleife
`r[4]` mutiert:

  * Produktions-/Gate-Fit (y_train)     -> nur 1.5x Gewicht
  * All-Data-Refit (liest r[4] spaeter) -> nur Label = Werbung
  * Tages-/Schattenserie (_build_train) -> beides

Deshalb reicht es nicht, das Gewicht abzuschalten oder das Label: `aus`
muss VOR der Mutation greifen, sonst traegt der Refit das Label weiter und
die Serie misst einen anderen Arm als ausgeliefert wird.
"""
import json
import re
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
QUELL = (REPO / "scripts/train-head.py").read_text()
REGEL = (REPO / "docs/o24-cluster-anker-preregistration.md").read_text()


def _anker_block():
    a = QUELL.index("cluster_anchored = r[13] if len(r) > 13 else []")
    e = QUELL.index("NaN-logo skip", a)
    return QUELL[a:e]


class ClusterAnker(unittest.TestCase):
    def test_boolmaske_kopiert(self):
        # Die Ursache der drei Varianten, als Tatsache festgehalten: wer das
        # "reparieren" will, muss wissen, dass es keine Referenz ist.
        y = np.zeros(4, dtype=np.int8)
        kopie = y[np.ones(4, dtype=bool)]
        y[1:3] = 1
        self.assertEqual(int(kopie.sum()), 0)

    def test_vorgabe_ist_aus_seit_o24(self):
        # O24 am 2026-09-23 NICHT ERFUELLT -> aus. Vorgabe UND Nightly, damit
        # Tagesserien und Handlaeufe nicht still auf dem alten Stand laufen.
        self.assertRegex(QUELL, r'"--cluster-anker", choices=\("alt", "aus"\), default="aus"')
        nightly = (REPO / "daemon/tv-train-head.sh").read_text()
        self.assertIn("    --cluster-anker aus \\\n", nightly)

    def test_anker_aus_dem_eigenen_verzeichnis(self):
        # Bis 2026-09-23 las die Zeile `rec_dir`, das diese Schleife nie
        # setzt: jede Aufnahme bekam die Anker der LETZTEN im Snapshot.
        self.assertIn('ca_path = (rec_dir_path / "cluster_anchored.json"', QUELL)
        self.assertNotIn('ca_path = rec_dir / "cluster_anchored.json"', QUELL)

    def test_aus_greift_vor_jeder_mutation(self):
        b = _anker_block()
        self.assertLess(b.index('args.cluster_anker == "aus"'),
                        b.index("yslice[i0:i1] = 1"),
                        "aus muss VOR der Label-Mutation greifen — sonst "
                        "traegt der Refit das Label weiter")
        self.assertLess(b.index('args.cluster_anker == "aus"'),
                        b.index("base_w * 1.5"))

    def test_nur_eine_anwendungsstelle(self):
        # Eine zweite Stelle (z. B. im Refit) wuerde der Schalter nicht sehen.
        self.assertEqual(QUELL.count("yslice[i0:i1] = 1"), 1)

    def test_einlesen_und_archiv_bleiben_unberuehrt(self):
        # Die Anker entscheiden mit, welche Aufnahmen ins Archiv kommen. Wuerde
        # `aus` schon das Einlesen abschalten, unterschieden sich die Arme im
        # KORPUS statt in der Behandlung.
        self.assertIn("or bool(cluster_anchored))):", QUELL)
        lese = QUELL.index('ca_path = (rec_dir_path / "cluster_anchored.json"')
        self.assertNotIn("cluster_anker", QUELL[lese - 800:lese + 400])

    def test_gewichte_tsv_zeigt_anker_je_aufnahme(self):
        # 2026-09-23: Log 97409 Anker-Frames, Snapshot hoechstens 71465.
        # Die Aufschluesselung je Aufnahme ist die einzige Adresse dafuer.
        self.assertIn("_anker_je_rec[r[0]] = (len(r[13])", QUELL)
        self.assertIn('f"\\t{_ns}\\t{_af}\\n"', QUELL)

    def test_gewichte_tsv_wird_jede_nacht_archiviert(self):
        # Die Bundle-Archivierung laeuft nur beim Deploy; die Kopie muss
        # direkt am Schreiber haengen, sonst fehlt die Vornacht bei REJECT.
        a = QUELL.index('.with_suffix(".gewichte.tsv")')
        self.assertIn('head.{ts}.gewichte.tsv', QUELL[a:a + 2500])

    def test_arme_stehen_in_registry_und_regel(self):
        m = re.search(r"```regel\n(.*?)\n```", REGEL, re.S)
        self.assertIsNotNone(m, "Regel-Block fehlt")
        regel = json.loads(m.group(1))
        for arm in regel["arme"].values():
            self.assertIn(f'_ts_arme["{arm}"] = (_ident, 32)', QUELL)


if __name__ == "__main__":
    unittest.main()
