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

    def test_vorgabe_bleibt_alt_bis_o24_entschieden(self):
        self.assertRegex(QUELL, r'"--cluster-anker", choices=\("alt", "aus"\), default="alt"')

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
        lese = QUELL.index('ca_path = rec_dir / "cluster_anchored.json"')
        self.assertNotIn("cluster_anker", QUELL[lese - 800:lese + 400])

    def test_arme_stehen_in_registry_und_regel(self):
        m = re.search(r"```regel\n(.*?)\n```", REGEL, re.S)
        self.assertIsNotNone(m, "Regel-Block fehlt")
        regel = json.loads(m.group(1))
        for arm in regel["arme"].values():
            self.assertIn(f'_ts_arme["{arm}"] = (_ident, 32)', QUELL)


if __name__ == "__main__":
    unittest.main()
