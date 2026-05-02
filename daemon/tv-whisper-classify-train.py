#!/usr/bin/env python3
"""
Retrain the whisper ad/show TF-IDF + LogReg classifier on the full
reviewed corpus (every recording with both ads_user.json AND a cached
.whisper.json).

A whisper window is labelled "ad" if its centre falls within any
user-confirmed ad block, else "show". Imbalanced corpus
(~22 % ad / 78 % show) handled by class_weight="balanced".

Usage:
    /Users/simon/ml/tv-classifier/.venv/bin/python3 \
        tv-whisper-classify-train.py [--dry-run]

Saves to ~/.cache/tv-whisper/classifier.pkl (overwrites; previous
bundle is rotated to .pkl.bak so a deploy can be undone in 1 step).
"""
import argparse
import json
import os
import pickle
import shutil
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold

GATEWAY = os.environ.get("GATEWAY", "https://raspberrypi5lan:8443")
CACHE = Path.home() / ".cache" / "tv-whisper"
CLASSIFIER_PATH = CACHE / "classifier.pkl"
WHISPER_WINDOW_S = 60
WHISPER_STRIDE_S = 30
CTX = ssl.create_default_context(); CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def fetch_snapshot():
    print("fetching training snapshot…", flush=True)
    return json.loads(urllib.request.urlopen(
        f"{GATEWAY}/api/internal/training-snapshot",
        context=CTX, timeout=60).read())


def build_corpus(snap, boundary_margin_s=30):
    """Walk reviewed recordings, label each whisper window. Skips
    windows whose 60-s span overlaps a block boundary (= label
    ambiguous: window straddles ad+show transition). Only purely-
    inside-ad or purely-outside-any-ad windows are kept. Default
    margin: drop windows whose centre is < boundary_margin_s from
    any block boundary."""
    texts, labels = [], []
    n_recs = n_skipped = n_ambiguous = 0
    for r in snap.get("recordings", []):
        uuid = r["uuid"]
        au = r.get("ads_user")
        if not au:
            continue
        blocks = au.get("ads") if isinstance(au, dict) else au
        if not blocks:
            continue
        jp = CACHE / f"{uuid}.whisper.json"
        if not jp.is_file():
            n_skipped += 1
            continue
        try:
            w = json.load(open(jp))
        except Exception:
            n_skipped += 1
            continue
        n_recs += 1
        for ww in w.get("windows", []):
            text = (ww.get("text") or "").strip()
            if len(text) < 10:
                continue
            t = float(ww["t"])
            centre = t + WHISPER_WINDOW_S / 2
            # Distance to nearest block boundary
            min_dist = min(
                min(abs(centre - s), abs(centre - e))
                for s, e in blocks)
            if min_dist < boundary_margin_s:
                n_ambiguous += 1
                continue
            in_ad = any(s <= centre <= e for s, e in blocks)
            texts.append(text)
            labels.append(1 if in_ad else 0)
    print(f"corpus: {n_recs} recordings used, {n_skipped} skipped, "
          f"{n_ambiguous} dropped as boundary-ambiguous")
    print(f"        {len(texts)} windows total "
          f"({sum(labels)} ad, {len(labels) - sum(labels)} show)")
    return texts, labels


def train_and_evaluate(texts, labels):
    print("vectorising…", flush=True)
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        lowercase=True,
        strip_accents="unicode",
    )
    X = vec.fit_transform(texts)
    print(f"vocab size: {len(vec.vocabulary_)}")
    print("5-fold CV…", flush=True)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=3.0,                        # tuned on n=18k corpus
        solver="liblinear",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        clf, X, labels, cv=cv,
        scoring=["precision", "recall", "f1", "accuracy"],
        n_jobs=-1)
    print(f"  precision: {scores['test_precision'].mean():.4f} "
          f"± {scores['test_precision'].std():.4f}")
    print(f"  recall:    {scores['test_recall'].mean():.4f} "
          f"± {scores['test_recall'].std():.4f}")
    print(f"  f1:        {scores['test_f1'].mean():.4f} "
          f"± {scores['test_f1'].std():.4f}")
    print(f"  accuracy:  {scores['test_accuracy'].mean():.4f} "
          f"± {scores['test_accuracy'].std():.4f}")
    print("fitting on full corpus…", flush=True)
    clf.fit(X, labels)
    bundle = {
        "vectorizer": vec,
        "classifier": clf,
        "n_train": len(texts),
        "cv_precision": float(scores["test_precision"].mean()),
        "cv_recall":    float(scores["test_recall"].mean()),
        "cv_f1":        float(scores["test_f1"].mean()),
        "cv_accuracy":  float(scores["test_accuracy"].mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "ngram_range": (1, 2),
        "min_df": 2,
    }
    return bundle


def save_bundle(bundle, dry_run=False):
    if dry_run:
        print("DRY-RUN: not writing classifier.pkl")
        return
    if CLASSIFIER_PATH.exists():
        bak = CLASSIFIER_PATH.with_suffix(".pkl.bak")
        shutil.copy(CLASSIFIER_PATH, bak)
        print(f"backup: {bak}")
    with open(CLASSIFIER_PATH, "wb") as fp:
        pickle.dump(bundle, fp)
    sz = CLASSIFIER_PATH.stat().st_size / 1024
    print(f"saved: {CLASSIFIER_PATH} ({sz:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="train + evaluate, don't overwrite classifier.pkl")
    args = ap.parse_args()
    t0 = time.time()
    snap = fetch_snapshot()
    texts, labels = build_corpus(snap)
    if len(texts) < 100:
        print(f"too few samples ({len(texts)}); aborting", file=sys.stderr)
        sys.exit(1)
    bundle = train_and_evaluate(texts, labels)
    save_bundle(bundle, dry_run=args.dry_run)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
