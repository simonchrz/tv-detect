#!/usr/bin/env python3
"""Aequivalenz-Tests fuer zusatzspalten() — die zusammengelegte Definition
des Zusatzblocks.

Bis 2026-08-07 stand dieselbe Rechnung an sechs Stellen. Dieser Test haelt
fest, dass die eine verbliebene Fassung EXAKT das liefert, was die sechs
alten geliefert haben — die alten Fassungen stehen unten woertlich als
Referenz, denn sie SIND die Spezifikation: jede Zahl, die je veroeffentlicht
wurde, ist mit ihnen entstanden.

⚠️ Verglichen wird bitgleich (array_equal), nicht allclose. Es ist dieselbe
Arithmetik in derselben Reihenfolge; ein Unterschied im letzten Bit waere
ein Hinweis darauf, dass eben NICHT dieselbe Rechnung laeuft, und genau das
soll der Test finden.

Ausfuehren: python3 scripts/test_zusatzspalten.py
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "th", str(Path(__file__).resolve().parent / "train-head.py"))
th = importlib.util.module_from_spec(_spec)
_argv = sys.argv
sys.argv = ["train-head"]
try:
    _spec.loader.exec_module(th)
except SystemExit:
    pass
finally:
    sys.argv = _argv


# ---- Umgebung: whisper + minute-prior deterministisch stellen --------------
#
# Beide Seiten (Referenz und neue Fassung) rufen dieselben Modulfunktionen,
# das Ersetzen hier wirkt also auf beide.

_WHISPER = {}      # uuid -> np.ndarray | None  (None = keine Whisper-Daten)
_MP = {}           # uuid -> np.ndarray (T,1)


def _stub_whisper_per_sec(uuid, n_seconds):
    a = _WHISPER.get(uuid)
    if a is None:
        return np.zeros(n_seconds, dtype=np.float32)
    out = np.zeros(n_seconds, dtype=np.float32)
    n = min(n_seconds, len(a))
    out[:n] = a[:n]
    return out


def _stub_whisper_present(uuid):
    return _WHISPER.get(uuid) is not None


th._load_whisper_per_sec = _stub_whisper_per_sec
th._whisper_present = _stub_whisper_present


def _mp_col(uuid, T):
    """Steht fuer _minuteprior_col — eine Funktion (uuid, T) -> (T,1)."""
    basis = _MP.get(uuid)
    if basis is None:
        return np.full((T, 1), 0.25, dtype=np.float32)
    out = np.zeros((T, 1), dtype=np.float32)
    n = min(T, len(basis))
    out[:n, 0] = basis[:n, 0]
    return out


# ---- Die sechs alten Fassungen, woertlich ---------------------------------


def alt_teacher(X, slug, chan_idx, uuid, wants_whisper, wants_temporal=False,
                mp_col=None, wants_churn=False, wants_mask=False):
    """_augment_teacher_feats, Stand vor der Zusammenlegung."""
    T = X.shape[0]
    oh = np.zeros((T, len(chan_idx)), dtype=np.float32)
    if slug in chan_idx:
        oh[:, chan_idx[slug]] = 1.0
    parts = [X, oh]
    if wants_whisper:
        parts.append(th._load_whisper_per_sec(uuid, T).reshape(-1, 1))
    if wants_temporal:
        dp = np.zeros((T, 1), dtype=np.float32)
        dn = np.zeros((T, 1), dtype=np.float32)
        if T > 1:
            d = np.linalg.norm(X[1:] - X[:-1], axis=1).astype(np.float32)
            dp[1:, 0] = d
            dn[:-1, 0] = d
        parts.append(dp)
        parts.append(dn)
        if wants_churn:
            parts.append(th._churn_col(X))
    if mp_col is not None:
        parts.append(mp_col(uuid, T))
    if wants_mask:
        parts.append(np.full((T, 1),
                             1.0 if th._whisper_present(uuid) else 0.0,
                             dtype=np.float32))
    return np.hstack(parts).astype(np.float32)


def alt_aug_test(X, slug, uuid, n_chan, prod_chan_idx, wants_whisper,
                 wants_temporal, wants_churn, wants_minuteprior,
                 wants_whispermask, mp_col):
    """_aug_test, Stand vor der Zusammenlegung. Gleiche Form wie der
    All-Data-Refit-Pfad (5090) und _augment_cwt_minuteprior."""
    T = X.shape[0]
    oh = np.zeros((T, n_chan), dtype=np.float32)
    if slug in prod_chan_idx:
        oh[:, prod_chan_idx[slug]] = 1.0
    parts = [X, oh]
    if wants_whisper:
        parts.append(th._load_whisper_per_sec(uuid, T).reshape(-1, 1))
    if wants_temporal:
        dp = np.zeros((T, 1), dtype=np.float32)
        dn = np.zeros((T, 1), dtype=np.float32)
        if T > 1:
            d = np.linalg.norm(X[1:] - X[:-1], axis=1).astype(np.float32)
            dp[1:, 0] = d
            dn[:-1, 0] = d
        parts.append(dp)
        parts.append(dn)
        if wants_churn:
            parts.append(th._churn_col(X))
    if wants_minuteprior:
        parts.append(mp_col(uuid, T))
    if wants_whispermask:
        parts.append(np.full(
            (T, 1), 1.0 if th._whisper_present(uuid) else 0.0,
            dtype=np.float32))
    return np.hstack(parts).astype(np.float32)


def alt_prod_blockweise(recs, masks, n_chan, chan_idx, wants_whisper,
                        wants_temporal, wants_churn, wants_minuteprior,
                        wants_whispermask, mp_col, basis_maskiert):
    """Der Produktions-Fit, Stand vor der Zusammenlegung: SPALTENBLOCK-weise
    ueber alle Aufnahmen, nicht pro Aufnahme.

    ⚠️ Das ist die Fassung, bei der die Zusammenlegung am ehesten schiefgeht:
    sie baut je Block eine Verkettung ueber alle Aufnahmen und haengt die
    Bloecke dann nebeneinander. Die neue Fassung baut pro Aufnahme und
    verkettet danach. Beide muessen dieselbe Matrix ergeben — das ist keine
    Selbstverstaendlichkeit, sondern haengt daran, dass die Zeilenreihenfolge
    in beiden Faellen die der Aufnahmen ist.
    """
    prod_parts = [basis_maskiert]
    chan_parts = []
    for (uuid, X, slug), mask in zip(recs, masks):
        T = X.shape[0]
        oh = np.zeros((T, n_chan), dtype=np.float32)
        if slug in chan_idx:
            oh[:, chan_idx[slug]] = 1.0
        chan_parts.append(oh[mask])
    prod_parts.append(np.concatenate(chan_parts))
    if wants_whisper:
        # ⚠️ WOERTLICH, einschliesslich des Fehlers: die Laenge ist die
        # NACH-Masken-Zahl n, und das Ergebnis wird NICHT maskiert. Alle
        # anderen Bloecke rechnen auf dem rohen T und maskieren danach.
        # Siehe MaskenVersatz unten.
        gesamt = sum(int(m.sum()) for m in masks)
        whisper_train = np.full(gesamt, 0.5, dtype=np.float32)
        offset = 0
        for (uuid, X, slug), mask in zip(recs, masks):
            n = int(mask.sum())
            if n <= 0:
                continue
            whisper_train[offset:offset + n] = th._load_whisper_per_sec(uuid, n)
            offset += n
        prod_parts.append(whisper_train.reshape(-1, 1))
    if wants_temporal:
        temporal_parts = []
        for (uuid, Xr, slug), mask in zip(recs, masks):
            Tr = Xr.shape[0]
            dp = np.zeros(Tr, dtype=np.float32)
            dn = np.zeros(Tr, dtype=np.float32)
            if Tr > 1:
                d = np.linalg.norm(
                    Xr[1:] - Xr[:-1], axis=1).astype(np.float32)
                dp[1:] = d
                dn[:-1] = d
            if wants_churn:
                ch = th._churn_col(Xr)[:, 0]
                temporal_parts.append(np.column_stack([dp, dn, ch])[mask])
            else:
                temporal_parts.append(np.column_stack([dp, dn])[mask])
        prod_parts.append(np.concatenate(temporal_parts))
    if wants_minuteprior:
        mp_parts = []
        for (uuid, X, slug), mask in zip(recs, masks):
            mp_parts.append(mp_col(uuid, X.shape[0])[mask])
        prod_parts.append(np.concatenate(mp_parts))
    if wants_whispermask:
        wm_parts = []
        for (uuid, X, slug), mask in zip(recs, masks):
            v = 1.0 if th._whisper_present(uuid) else 0.0
            wm_parts.append(
                np.full((X.shape[0], 1), v, dtype=np.float32)[mask])
        prod_parts.append(np.concatenate(wm_parts))
    return np.hstack(prod_parts)


def alt_augment_channel(X, slug, n_chan, chan_idx):
    T = X.shape[0]
    oh = np.zeros((T, n_chan), dtype=np.float32)
    if slug in chan_idx:
        oh[:, chan_idx[slug]] = 1.0
    return np.hstack([X, oh])


def alt_augment_temporal(X):
    T = X.shape[0]
    dp = np.zeros(T, dtype=np.float32)
    dn = np.zeros(T, dtype=np.float32)
    if T > 1:
        d = np.linalg.norm(X[1:] - X[:-1], axis=1).astype(np.float32)
        dp[1:] = d
        dn[:-1] = d
    return np.column_stack([X, dp, dn]).astype(np.float32)


# ---- Testdaten ------------------------------------------------------------

SLUGS = ["ard", "prosieben", "rtl", "vox"]
CHAN_IDX = {s: i for i, s in enumerate(SLUGS)}


def _mach_aufnahme(rng, uuid, T, breite=12, slug="rtl", mit_whisper=True):
    X = rng.standard_normal((T, breite)).astype(np.float32)
    if mit_whisper:
        _WHISPER[uuid] = rng.random(T).astype(np.float32)
    else:
        _WHISPER.pop(uuid, None)
    _MP[uuid] = rng.random((T, 1)).astype(np.float32)
    return uuid, X, slug


# alle Flaggen-Kombinationen, die im Code vorkommen (arch v1..v5)
ARCHEN = [
    dict(whisper=False, temporal=False, churn=False, mp=False, maske=False),
    dict(whisper=True,  temporal=False, churn=False, mp=False, maske=False),
    dict(whisper=True,  temporal=True,  churn=False, mp=False, maske=False),
    dict(whisper=True,  temporal=True,  churn=False, mp=True,  maske=False),
    dict(whisper=True,  temporal=True,  churn=False, mp=True,  maske=True),
    dict(whisper=True,  temporal=True,  churn=True,  mp=True,  maske=True),
]


class GleichWieAlt(unittest.TestCase):

    def _faelle(self):
        rng = np.random.default_rng(7)
        # T-Werte inkl. der Randfaelle, an denen die alte Fassung schon
        # einmal gestorben ist (T kuerzer als das Unruhe-Fenster).
        for T in (1, 2, 5, 60, 61, 62, 200):
            for slug in ("rtl", "nicht-im-korpus"):
                for mit_whisper in (True, False):
                    uuid = f"u-{T}-{slug}-{mit_whisper}"
                    yield _mach_aufnahme(rng, uuid, T, slug=slug,
                                         mit_whisper=mit_whisper)

    def test_gegen_lehrer(self):
        for uuid, X, slug in self._faelle():
            for a in ARCHEN:
                with self.subTest(T=X.shape[0], slug=slug, arch=a):
                    alt = alt_teacher(
                        X, slug, CHAN_IDX, uuid,
                        wants_whisper=a["whisper"],
                        wants_temporal=a["temporal"],
                        mp_col=_mp_col if a["mp"] else None,
                        wants_churn=a["churn"], wants_mask=a["maske"])
                    neu = th.mit_zusatz(
                        X, uuid, slug, CHAN_IDX,
                        whisper=a["whisper"], temporal=a["temporal"],
                        churn=a["churn"],
                        mp_col=_mp_col if a["mp"] else None,
                        maske=a["maske"])
                    self.assertTrue(np.array_equal(alt, neu),
                                    f"Lehrer-Pfad weicht ab: {alt.shape} "
                                    f"gegen {neu.shape}")

    def test_gegen_aug_test(self):
        n_chan = len(SLUGS)
        for uuid, X, slug in self._faelle():
            for a in ARCHEN:
                with self.subTest(T=X.shape[0], slug=slug, arch=a):
                    alt = alt_aug_test(
                        X, slug, uuid, n_chan, CHAN_IDX,
                        a["whisper"], a["temporal"], a["churn"],
                        a["mp"], a["maske"], _mp_col)
                    neu = th.mit_zusatz(
                        X, uuid, slug, CHAN_IDX, n_chan,
                        whisper=a["whisper"], temporal=a["temporal"],
                        churn=a["churn"],
                        mp_col=_mp_col if a["mp"] else None,
                        maske=a["maske"])
                    self.assertTrue(np.array_equal(alt, neu))

    def _korpus(self, luecken=True):
        rng = np.random.default_rng(11)
        recs = [_mach_aufnahme(rng, "p1", 90, slug="rtl"),
                _mach_aufnahme(rng, "p2", 61, slug="ard",
                               mit_whisper=False),
                _mach_aufnahme(rng, "p3", 7, slug="nicht-im-korpus"),
                _mach_aufnahme(rng, "p4", 1, slug="vox")]
        if luecken:
            # Hygiene-Masken: loechrig, unterschiedlich dicht — so sieht es
            # nach einem Hygiene-Durchlauf aus.
            masks = [rng.random(X.shape[0]) > 0.3 for _, X, _ in recs]
            masks[1][:] = True
            masks[3][:] = True
        else:
            masks = [np.ones(X.shape[0], bool) for _, X, _ in recs]
        return recs, masks

    def _neu(self, recs, masks, a, n_chan):
        basis = np.concatenate([X[m] for (_, X, _), m in zip(recs, masks)])
        zusatz = np.concatenate([
            th.zusatzspalten(
                X, uuid, slug, CHAN_IDX, n_chan,
                whisper=a["whisper"], temporal=a["temporal"],
                churn=a["churn"], mp_col=_mp_col if a["mp"] else None,
                maske=a["maske"])[m]
            for (uuid, X, slug), m in zip(recs, masks)])
        return basis, np.hstack([basis, zusatz])

    def test_gegen_produktions_fit_ohne_luecken(self):
        """Blockweise ueber alle Aufnahmen gegen aufnahmeweise mit
        anschliessender Verkettung — bei VOLLEN Masken bitgleich.

        Das ist der Fall, der beweist, dass die Umstellung von
        'Spaltenblock ueber alle Aufnahmen' auf 'pro Aufnahme, dann
        verketten' die Matrix nicht antastet."""
        recs, masks = self._korpus(luecken=False)
        n_chan = len(SLUGS)
        for a in ARCHEN:
            with self.subTest(arch=a):
                basis, neu = self._neu(recs, masks, a, n_chan)
                alt = alt_prod_blockweise(
                    recs, masks, n_chan, CHAN_IDX,
                    a["whisper"], a["temporal"], a["churn"],
                    a["mp"], a["maske"], _mp_col, basis)
                self.assertEqual(alt.shape, neu.shape)
                self.assertTrue(np.array_equal(alt, neu))

    def test_gegen_produktions_fit_alles_ausser_whisper(self):
        """Mit loechrigen Masken stimmt alles ueberein — ausser der
        Whisper-Spalte. Siehe MaskenVersatz."""
        recs, masks = self._korpus(luecken=True)
        n_chan = len(SLUGS)
        for a in ARCHEN:
            with self.subTest(arch=a):
                basis, neu = self._neu(recs, masks, a, n_chan)
                alt = alt_prod_blockweise(
                    recs, masks, n_chan, CHAN_IDX,
                    a["whisper"], a["temporal"], a["churn"],
                    a["mp"], a["maske"], _mp_col, basis)
                self.assertEqual(alt.shape, neu.shape)
                if not a["whisper"]:
                    self.assertTrue(np.array_equal(alt, neu))
                    continue
                w = basis.shape[1] + n_chan       # Spalte der Whisper-Werte
                ohne = [c for c in range(alt.shape[1]) if c != w]
                self.assertTrue(np.array_equal(alt[:, ohne], neu[:, ohne]))


    def test_gegen_schatten_varianten(self):
        """Die --shadow-eval-Sonden v1/v2/v3 — v3 hat KEIN one-hot."""
        rng = np.random.default_rng(3)
        n_chan = len(SLUGS)
        for T in (1, 2, 40):
            uuid, X, slug = _mach_aufnahme(rng, f"s{T}", T, slug="ard")
            self.assertTrue(np.array_equal(
                alt_augment_channel(X, slug, n_chan, CHAN_IDX),
                th.mit_zusatz(X, uuid, slug, CHAN_IDX, n_chan)))
            self.assertTrue(np.array_equal(
                alt_augment_temporal(X),
                th.mit_zusatz(X, uuid, slug, CHAN_IDX, n_chan,
                              kanal=False, temporal=True)))
            # v1 = gar keine Anreicherung
            self.assertTrue(np.array_equal(
                X, th.mit_zusatz(X, uuid, slug, CHAN_IDX, n_chan,
                                 kanal=False)))



class MaskenVersatz(unittest.TestCase):
    """⚠️ Ein echter Fehler, gefunden BEIM Zusammenlegen (2026-08-07).

    Der Produktions-Fit rief `_load_whisper_per_sec(uuid, n)` mit der
    NACH-Masken-Frame-Zahl auf und schrieb das Ergebnis ungefiltert in den
    maskierten Bereich. Die Funktion indiziert Whisper-Fenster nach
    ABSOLUTER Sekunde und schneidet bei n ab — die Spalte war also um die
    Zahl der bis dahin verworfenen Frames verschoben, und das Ende der
    Aufnahme fiel ganz heraus.

    Alle anderen Bloecke (temporal, minute-prior, maske) rechnen auf dem
    rohen T und maskieren danach; die Auswertung (_aug_test) und die
    Go-Seite ebenfalls. Betroffen war also ausschliesslich das TRAINING,
    und dort im letzten Lauf 168 von 283 Aufnahmen.

    Der Test haelt beides fest: dass der alte Weg wirklich verschoben war,
    und dass der neue mit der Auswertung uebereinstimmt.
    """

    def test_alter_weg_war_verschoben(self):
        T, drops = 600, 40
        rng = np.random.default_rng(5)
        mask = np.ones(T, bool)
        mask[rng.choice(T, drops, replace=False)] = False
        n = int(mask.sum())
        # Whisper-Wert = absolute Sekunde, damit der Versatz ablesbar ist
        _WHISPER["v"] = np.arange(T, dtype=np.float32)
        alt = th._load_whisper_per_sec("v", n)              # alter Weg
        neu = th._load_whisper_per_sec("v", T)[mask]        # neuer Weg
        versatz = neu - alt
        # Der Versatz ist die Zahl der bis dahin verworfenen Sekunden: er
        # waechst monoton und ist am Ende so gross wie die Zahl der Luecken
        # vor dem letzten ueberlebenden Frame. Er beginnt NICHT zwingend bei
        # null — faellt schon Sekunde 0 weg, ist er ab dem ersten Frame >0.
        self.assertTrue(np.all(np.diff(versatz) >= 0), "nicht monoton")
        letzter = int(np.flatnonzero(mask)[-1])
        self.assertEqual(float(versatz[-1]),
                         float(drops - (T - 1 - letzter)))
        self.assertGreater(float(versatz[-1]), 0.0)
        # Und das Ende der Aufnahme kommt im Training gar nicht vor: der
        # alte Weg sieht keine Sekunde jenseits von n-1.
        self.assertEqual(float(alt.max()), float(n - 1))
        self.assertEqual(float(neu.max()), float(letzter))
        self.assertGreater(float(neu.max()), float(alt.max()))

    def test_neu_stimmt_mit_der_auswertung_ueberein(self):
        """Der Fit sieht jetzt dieselbe Spalte wie die Auswertung — das ist
        der eigentliche Punkt der Uebung."""
        T, drops = 600, 40
        rng = np.random.default_rng(6)
        mask = np.ones(T, bool)
        mask[rng.choice(T, drops, replace=False)] = False
        X = rng.standard_normal((T, 8)).astype(np.float32)
        _WHISPER["w"] = rng.random(T).astype(np.float32)
        _MP["w"] = rng.random((T, 1)).astype(np.float32)
        fit = th.zusatzspalten(X, "w", "rtl", CHAN_IDX, len(SLUGS),
                               whisper=True, temporal=True, churn=True,
                               mp_col=_mp_col, maske=True)[mask]
        auswertung = alt_aug_test(X, "rtl", "w", len(SLUGS), CHAN_IDX,
                                  True, True, True, True, True,
                                  _mp_col)[mask][:, X.shape[1]:]
        self.assertTrue(np.array_equal(fit, auswertung))

class Breitenvertrag(unittest.TestCase):
    """Die Blockbreite MUSS die sein, die der Header verspricht — der Kopf
    prueft sie beim Laden, und eine falsche Breite faellt dort als
    'unalignable' auf und schaltet den betroffenen Pfad still ab."""

    def _breite(self, **kw):
        rng = np.random.default_rng(1)
        uuid, X, slug = _mach_aufnahme(rng, "b", 40, slug="rtl")
        return th.zusatzspalten(X, uuid, slug, CHAN_IDX, **kw).shape[1]

    def test_breiten_je_arch(self):
        nc = len(SLUGS)
        self.assertEqual(self._breite(), nc)                       # v1+chan
        self.assertEqual(self._breite(whisper=True), nc + 1)       # v2
        self.assertEqual(self._breite(whisper=True, temporal=True),
                         nc + 3)                                   # v3
        self.assertEqual(self._breite(whisper=True, temporal=True,
                                      mp_col=_mp_col), nc + 4)     # v4
        self.assertEqual(self._breite(whisper=True, temporal=True,
                                      mp_col=_mp_col, maske=True),
                         nc + 5)                                   # v5
        self.assertEqual(self._breite(whisper=True, temporal=True,
                                      churn=True, mp_col=_mp_col,
                                      maske=True), nc + 6)         # v5+churn

    def test_unruhe_ohne_temporal_ist_wirkungslos(self):
        nc = len(SLUGS)
        self.assertEqual(self._breite(whisper=True, churn=True), nc + 1)

    def test_leerer_block(self):
        rng = np.random.default_rng(2)
        uuid, X, slug = _mach_aufnahme(rng, "l", 10)
        z = th.zusatzspalten(X, uuid, slug, CHAN_IDX, kanal=False)
        self.assertEqual(z.shape, (10, 0))
        self.assertTrue(np.array_equal(
            th.mit_zusatz(X, uuid, slug, CHAN_IDX, kanal=False), X))

    def test_unbekannter_slug_ist_nullvektor(self):
        """Kein Absturz, keine falsche Spalte — ein Kanal ausserhalb der
        Karte bekommt ein leeres one-hot. Der Kopf faellt damit auf
        'kanalagnostisch' zurueck, was die dokumentierte Absicht ist."""
        rng = np.random.default_rng(4)
        uuid, X, slug = _mach_aufnahme(rng, "x", 20, slug="gibts-nicht")
        z = th.zusatzspalten(X, uuid, slug, CHAN_IDX)
        self.assertEqual(z.shape, (20, len(SLUGS)))
        self.assertEqual(float(z.sum()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
