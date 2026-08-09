#!/usr/bin/env python3
"""Tests fuer den versiegelten Satz.

Nachgebaut statt importiert: die Logik sitzt mitten in main() von
train-head.py und ist von aussen nicht aufrufbar. Der Test haelt deshalb eine
Kopie der ENTSCHEIDUNGSREGEL und prueft, dass die Kopie und das Original
dieselben uuids versiegeln — plus die Eigenschaften, an denen alles haengt.

⚠️ Der Punkt ist nicht, dass versiegelt wird. Der Punkt ist, dass NICHTS
versiegelt wird, was schon im Ledger steht: eine Aufnahme nachtraeglich zu
versiegeln, auf der schon trainiert wurde, macht den Satz wertlos und es
faellt niemandem auf — die Zahl sieht dann nur etwas besser aus.

Ausfuehren: python3 scripts/test_versiegelter_satz.py
"""
import hashlib
import re
import unittest
from pathlib import Path

QUELLE = Path(__file__).resolve().parent / "train-head.py"
VERSIEGELT = "versiegelt"


def regel(uuid_str, ledger, *, seeding, frac, golden=frozenset(),
          ausgeschlossen=frozenset()):
    """Kopie der Regel aus train-head.py. Gibt (versiegelt?, ledger)."""
    if ledger.get(uuid_str) == VERSIEGELT:
        return True
    if uuid_str in ledger:
        return False
    if seeding or frac <= 0:
        return False
    if uuid_str in golden or uuid_str in ausgeschlossen:
        return False
    h = int(hashlib.md5(("versiegelt:" + uuid_str).encode()).hexdigest(), 16)
    if h / 2**128 >= frac:
        return False
    ledger[uuid_str] = VERSIEGELT
    return True


class RuehrtBestehendesNichtAn(unittest.TestCase):

    def test_bekannte_uuid_wird_nie_versiegelt(self):
        # Die wichtigste Eigenschaft ueberhaupt. Eine Aufnahme, auf der schon
        # trainiert wurde, nachtraeglich zu versiegeln macht den Satz wertlos.
        for eimer in ("train", "test"):
            for i in range(200):
                u = f"alt-{i}"
                led = {u: eimer}
                self.assertFalse(regel(u, led, seeding=False, frac=1.0),
                                 f"{u} ({eimer}) wurde versiegelt")
                self.assertEqual(led[u], eimer)

    def test_erstbefuellung_versiegelt_nichts(self):
        # Bei leerem Ledger ist JEDE uuid neu — ohne diese Bremse verschwaende
        # ein einziger Lauf ein Fuenftel des Korpus.
        led = {}
        n = sum(regel(f"u-{i}", led, seeding=True, frac=0.2) for i in range(500))
        self.assertEqual(n, 0)
        self.assertEqual(led, {})

    def test_aus_bedeutet_aus(self):
        led = {"vorhanden": "train"}
        n = sum(regel(f"u-{i}", led, seeding=False, frac=0.0)
                for i in range(500))
        self.assertEqual(n, 0)

    def test_golden_pins_bleiben_unversiegelt(self):
        led = {"vorhanden": "train"}
        golden = {f"g-{i}" for i in range(200)}
        n = sum(regel(u, led, seeding=False, frac=1.0, golden=golden)
                for u in sorted(golden))
        self.assertEqual(n, 0)


class Anteil(unittest.TestCase):

    def test_anteil_stimmt_ungefaehr(self):
        led = {"vorhanden": "train"}
        n = sum(regel(f"neu-{i}", led, seeding=False, frac=0.2)
                for i in range(4000))
        self.assertGreater(n, 4000 * 0.17)
        self.assertLess(n, 4000 * 0.23)

    def test_entscheidung_ist_stabil(self):
        # Zweimal gefragt, zweimal dieselbe Antwort — sonst wanderte die
        # Zugehoerigkeit von Nacht zu Nacht und der Satz waere kein Satz.
        led = {"vorhanden": "train"}
        erste = [regel(f"neu-{i}", led, seeding=False, frac=0.2)
                 for i in range(500)]
        zweite = [regel(f"neu-{i}", led, seeding=False, frac=0.2)
                  for i in range(500)]
        self.assertEqual(erste, zweite)

    def test_eigener_salt_entkoppelt_vom_test_split(self):
        # Gleicher Hash wie der Test-Split waere eine Korrelation zwischen
        # "versiegelt" und "im Testsatz" — der versiegelte Satz waere dann
        # systematisch anders zusammengesetzt als der Rest des Korpus.
        def test_split_hash(u):
            return int(hashlib.md5(u.encode()).hexdigest(), 16) / 2**128

        led = {"vorhanden": "train"}
        versiegelt, im_test = [], []
        for i in range(3000):
            u = f"neu-{i}"
            versiegelt.append(regel(u, dict(led), seeding=False, frac=0.2))
            im_test.append(test_split_hash(u) < 0.2)
        beide = sum(1 for a, b in zip(versiegelt, im_test) if a and b)
        # Bei Unabhaengigkeit ~0.04 der Faelle; bei identischem Hash ~0.20.
        self.assertLess(beide / 3000, 0.08)


class PasstZumOriginal(unittest.TestCase):

    def test_quelle_nutzt_denselben_salt_und_dieselbe_reihenfolge(self):
        src = QUELLE.read_text()
        self.assertIn('"versiegelt:" + uuid_str', src,
                      "Salt im Original geaendert — Zugehoerigkeiten wandern")
        # _is_sealed MUSS vor _is_test stehen, sonst schreibt _is_test beim
        # ersten Sehen einen Eimer und die uuid ist nie mehr neu.
        m = re.search(r"train_recs = \[r for r in per_rec\s*\n\s*if (.+?)\n",
                      src)
        self.assertIsNotNone(m, "train_recs-Zeile nicht gefunden")
        self.assertIn("_is_sealed", m.group(1))
        self.assertLess(m.group(1).index("_is_sealed"),
                        m.group(1).index("_is_test")
                        if "_is_test" in m.group(1) else 10**6)
        # Und der Testsatz muss ebenfalls filtern, sonst wird gegen den
        # versiegelten Satz ausgewertet — genau das, was er nicht sein soll.
        m2 = re.search(r"test_recs\s+= \[r for r in per_rec\s*\n\s*if (.+?)\n",
                       src)
        self.assertIsNotNone(m2, "test_recs-Zeile nicht gefunden")
        self.assertIn("_is_sealed", m2.group(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
