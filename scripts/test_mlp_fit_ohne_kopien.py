#!/usr/bin/env python3
"""Der kopienfreie Fit muss bitgleich dasselbe lernen wie die kopierende Fassung.

Bis 2026-09-16 kopierte WeightedMLP.fit die Trainingsmatrix bis zu zweimal
vollstaendig (`X[keep]`, `X[perm]`) -- je Fit, dreimal pro Nacht und noch
einmal fuer die Voll-Anpassung, bei 12 GB je Kopie. Das hat den Nightly in
der Nacht zum 16.09. ueber 64 GB gedrueckt (SIGKILL vor dem Gate).

Die neue Fassung sortiert nur Zeilen-INDIZES. Dass sie dieselben Zahlen
liefert, ist hier gegen die alte Fassung belegt, die WOERTLICH aus dem
Commit vor dem Umbau eingefroren ist -- nicht gegen eine Erinnerung daran.
"""
import hashlib
import re
import unittest
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
NEU_SRC = (HIER / "train-head.py").read_text()

ALT_KLASSE = r'''class WeightedMLP:
    """Single-hidden-layer MLP (relu → sigmoid) with true fractional
    sample_weight — the drop-in replacement for sklearn's MLPClassifier
    in all head fits (2026-07-07).

    Why not MLPClassifier: it has no sample_weight, so weights were
    approximated by integer-rounded row duplication (max(round(sw),1)).
    That quantization silently erased the whole designed weighting —
    pseudo 0.3→1 (3× too strong), age-decay 0.5..0.99→1, bumper-boost
    1.4→1, NaN-logo 0→1 — only the user 2× survived. It also cost two
    full oversampled matrix copies (~29 GB peak) per fit.

    Semantics kept from MLPClassifier: Glorot-uniform init, Adam
    (lr 1e-3, β 0.9/0.999), L2 alpha 1e-4, early stopping on a 10%
    validation split with patience 10 / tol 1e-4, best-epoch weights
    restored. Deliberate differences: batch 512 (was min(200,n)) for
    BLAS efficiency, validation criterion = weighted log-loss (was
    accuracy), rows with weight<=0 are dropped up front. Deterministic
    per (data, random_state) like before. Exposes coefs_/intercepts_/
    n_iter_/loss_/predict/predict_proba, so write_mlp_head_v1/v2 and
    every consumer stay unchanged."""

    def __init__(self, hidden_dim=32, max_iter=80, random_state=0,
                 alpha=1e-4, batch_size=512, lr=1e-3,
                 validation_fraction=0.1, n_iter_no_change=10, tol=1e-4):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.batch_size = batch_size
        self.lr = lr
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.coefs_ = None
        self.intercepts_ = None
        self.n_iter_ = 0
        self.loss_ = float("nan")

    def fit(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        w = (np.ones(len(y), dtype=np.float32) if sample_weight is None
             else np.asarray(sample_weight, dtype=np.float32).ravel())
        keep = w > 0
        if not keep.all():
            X, y, w = X[keep], y[keep], w[keep]
        n = len(y)
        if n == 0:
            raise ValueError("WeightedMLP.fit: no rows with weight > 0")
        # One shuffled copy; train/val are then contiguous slices (views),
        # so peak memory is input + this copy — no oversample, no second
        # sklearn-internal split copy.
        perm = rng.permutation(n)
        X, y, w = X[perm], y[perm], w[perm]
        n_val = max(1, int(n * self.validation_fraction)) if n >= 10 else 0
        Xt, yt, wt = X[n_val:], y[n_val:], w[n_val:]
        Xv, yv, wv = X[:n_val], y[:n_val], w[:n_val]

        d, h = X.shape[1], self.hidden_dim
        bound1 = np.sqrt(6.0 / (d + h))
        bound2 = np.sqrt(6.0 / (h + 1))
        W1 = rng.uniform(-bound1, bound1, (d, h)).astype(np.float32)
        b1 = np.zeros(h, dtype=np.float32)
        W2 = rng.uniform(-bound2, bound2, (h, 1)).astype(np.float32)
        b2 = np.zeros(1, dtype=np.float32)
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0

        def _val_loss():
            if n_val == 0:
                return float("nan")
            p = self._forward_params(Xv, W1, b1, W2, b2)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            bce = -(yv * np.log(p) + (1 - yv) * np.log(1 - p))
            return float((bce * wv).sum() / wv.sum())

        best_val = np.inf
        best_params = None
        stale = 0
        nt = len(yt)
        for epoch in range(1, self.max_iter + 1):
            order = rng.permutation(nt)
            epoch_loss = 0.0
            epoch_wsum = 0.0
            for lo in range(0, nt, self.batch_size):
                idx = order[lo:lo + self.batch_size]
                xb, yb, wb = Xt[idx], yt[idx], wt[idx]
                nb = len(idx)
                z1 = xb @ W1 + b1
                a1 = np.maximum(z1, 0.0)
                z2 = (a1 @ W2).ravel() + b2[0]
                p = 1.0 / (1.0 + np.exp(-z2))
                pc = np.clip(p, 1e-7, 1 - 1e-7)
                wsum = wb.sum()
                bce = -(yb * np.log(pc) + (1 - yb) * np.log(1 - pc))
                epoch_loss += float((bce * wb).sum())
                epoch_wsum += float(wsum)
                # weighted BCE gradient + sklearn-style L2 (alpha/batch)
                dz2 = ((p - yb) * wb / wsum).astype(np.float32)
                gW2 = a1.T @ dz2[:, None] + (self.alpha / nb) * W2
                gb2 = np.array([dz2.sum()], dtype=np.float32)
                da1 = dz2[:, None] @ W2.T
                dz1 = da1 * (z1 > 0)
                gW1 = xb.T @ dz1 + (self.alpha / nb) * W1
                gb1 = dz1.sum(axis=0)
                t_step += 1
                for pi, gi in zip(range(4), (gW1, gb1, gW2, gb2)):
                    m[pi] = beta1 * m[pi] + (1 - beta1) * gi
                    v[pi] = beta2 * v[pi] + (1 - beta2) * gi * gi
                    mh = m[pi] / (1 - beta1 ** t_step)
                    vh = v[pi] / (1 - beta2 ** t_step)
                    params[pi] -= (self.lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
                W1, b1, W2, b2 = params
            self.n_iter_ = epoch
            self.loss_ = epoch_loss / max(epoch_wsum, 1e-9)
            vl = _val_loss()
            if n_val:
                if vl < best_val - self.tol:
                    best_val = vl
                    best_params = [p.copy() for p in params]
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.n_iter_no_change:
                        break
        if best_params is not None:
            W1, b1, W2, b2 = best_params
        self.coefs_ = [W1.astype(np.float64), W2.astype(np.float64)]
        self.intercepts_ = [b1.astype(np.float64), b2.astype(np.float64)]
        return self

    @staticmethod
    def _forward_params(X, W1, b1, W2, b2, chunk=1 << 18):
        out = np.empty(len(X), dtype=np.float64)
        for lo in range(0, len(X), chunk):
            xb = np.asarray(X[lo:lo + chunk], dtype=np.float32)
            a1 = np.maximum(xb @ W1 + b1, 0.0)
            z2 = (a1 @ W2).ravel() + b2[0]
            out[lo:lo + chunk] = 1.0 / (1.0 + np.exp(-z2.astype(np.float64)))
        return out

    def predict_proba(self, X):
        p = self._forward_params(X, self.coefs_[0].astype(np.float32),
                                 self.intercepts_[0].astype(np.float32),
                                 self.coefs_[1].astype(np.float32),
                                 self.intercepts_[1].astype(np.float32))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


'''


def _klasse_aus(src):
    """Klassentext herausschneiden — aus der ganzen Datei ODER aus der
    eingefrorenen Referenz, die direkt mit `class` beginnt."""
    m0 = re.search(r"(?m)^class WeightedMLP\b", src)
    i = m0.start()
    m1 = re.search(r"(?m)^(?:def |class )", src[i + 1:])
    ende = i + 1 + m1.start() if m1 else len(src)
    ns = {"np": np}
    exec(src[i:ende], ns)
    return ns["WeightedMLP"]


Alt = _klasse_aus(ALT_KLASSE)
Neu = _klasse_aus(NEU_SRC)


def _daten(n, d, nullen, seed):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.float32)
    w = r.uniform(0.2, 2.0, n).astype(np.float32)
    if nullen:
        w[r.choice(n, n // 20, replace=False)] = 0.0   # wie Logo-NaN-Frames
    return X, y, w


def _gleich(a, b):
    return (all(np.array_equal(p, q) for p, q in zip(a.coefs_, b.coefs_))
            and all(np.array_equal(p, q) for p, q in zip(a.intercepts_, b.intercepts_))
            and a.n_iter_ == b.n_iter_ and a.loss_ == b.loss_)


class FitOhneKopienIstBitgleich(unittest.TestCase):
    def _vergleich(self, n, d, nullen, seed=7):
        X, y, w = _daten(n, d, nullen, seed)
        a = Alt(hidden_dim=8, random_state=3, max_iter=12).fit(X, y, w)
        b = Neu(hidden_dim=8, random_state=3, max_iter=12).fit(X, y, w)
        self.assertTrue(_gleich(a, b), f"n={n} nullen={nullen}: Koeffizienten weichen ab")
        # und die Vorhersage auf fremden Daten ebenso
        Xn = np.random.default_rng(99).normal(size=(500, d)).astype(np.float32)
        self.assertTrue(np.array_equal(a.predict_proba(Xn), b.predict_proba(Xn)))

    def test_mit_nullgewichten(self):
        self._vergleich(6000, 40, nullen=True)

    def test_ohne_nullgewichte(self):
        self._vergleich(6000, 40, nullen=False)

    def test_winzig_ohne_validierung(self):
        # n < 10: n_val = 0, der Validierungspfad ist aus
        self._vergleich(8, 5, nullen=False)

    def test_eingabe_bleibt_unangetastet(self):
        X, y, w = _daten(3000, 20, True, 1)
        X0 = X.copy()
        Neu(hidden_dim=8, random_state=3, max_iter=5).fit(X, y, w)
        self.assertTrue(np.array_equal(X, X0))


class KeineVollkopieMehrImQuelltext(unittest.TestCase):
    def test_die_beiden_kopien_sind_weg(self):
        i = NEU_SRC.index("\nclass WeightedMLP")
        j = NEU_SRC.index("\ndef merge_mlp_ensemble")
        fit = NEU_SRC[i:j]
        self.assertNotIn("X, y, w = X[keep], y[keep], w[keep]", fit)
        self.assertNotIn("X, y, w = X[perm], y[perm], w[perm]", fit)
        self.assertIn("X[rows_t[idx]]", fit)


class FingerabdruckOhneByteKopie(unittest.TestCase):
    def test_blockweise_gleicher_digest(self):
        r = np.random.default_rng(5)
        for a in (r.normal(size=(70001, 37)).astype(np.float32),
                  r.normal(size=200003).astype(np.float32),
                  r.integers(0, 2, 12345).astype(np.float32)):
            voll = hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()
            h = hashlib.sha1()
            for lo in range(0, len(a), 65536):
                h.update(np.ascontiguousarray(a[lo:lo + 65536]))
            self.assertEqual(voll, h.hexdigest())

    def test_train_head_hasht_blockweise(self):
        i = NEU_SRC.index("def _fp(a):")
        # Nur Code zaehlt — der Kommentar darueber nennt `.tobytes()` ja
        # gerade als das, was vermieden wird.
        code = "\n".join(z for z in NEU_SRC[i:i + 900].splitlines()
                         if not z.lstrip().startswith("#"))
        self.assertNotIn(".tobytes()", code)
        self.assertIn("h.update(", code)


if __name__ == "__main__":
    unittest.main()
