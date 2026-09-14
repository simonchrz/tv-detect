#!/usr/bin/env python3
"""Ausschlussliste fuer den Massstab: EINE Definition fuer alle Leser.

Bis 2026-09-14 lebte TEST_SET_EXCLUDE nur in train-head.py. Der Trainer
hielt sich daran -- golden_v3_vorschlag.py und massstab-audit.py leiteten
den "Test-Eimer" aber aus dem Split-Ledger ab, das die Liste nicht kennt.
Folge: 12 von 113 Ledger-Test-Aufnahmen standen auf der Ausschlussliste
oder in der Quarantaene und wurden trotzdem als Massstab gezaehlt; der
Review-Hebel bot drei davon als Golden-Kandidaten an, darunter eine
Let's-Dance-Aufnahme, die Simon am 07.09. ausdruecklich quarantaeniert
hatte. Der Golden-Satz selbst war sauber (0 Ueberschneidungen).

Dieselbe Bauform wie label_herkunft.py: wer die Liste aendert, aendert sie
fuer alle drei Leser. train-head.py laedt dieses Modul per importlib.
"""
from pathlib import Path

# TEST_SET_EXCLUDE — known-bad ground truth, never trust as a test
# target (still eligible for train_recs at its normal which=auto
# weight; only the SCORING role is revoked). Root-caused 2026-07-13:
# "Reisen mit Kreta.de" sat at IoU 0.00 for the whole shadow-eval
# week, dragging the movies/niche median down. Its "ads" field
# (which=auto, never user-reviewed) has exactly ONE block
# (414-582s), but the SAME recording's own cluster_anchored list
# independently flags 9 more high-confidence ad spots (family sizes
# 3-59, i.e. matched against 3-59 other airings) between 618s and
# 1029s that never made it into "ads" — the auto-cutlist truth is
# badly incomplete, so the model's low IoU against it was punishing
# correct detections, not revealing a real weakness. Flagged in the
# 2026-07-07 optimization backlog review; add future confirmed-bad
# test recs here rather than re-litigating per rejection.
TEST_SET_EXCLUDE = {
    "dvr-anixe-1781518500",
    # 2026-07-30 label audit: frozen-archive entry (source long gone,
    # serien-retention) with a 361s hole on user labels — NN sure-ad
    # inside labelled show, unverifiable forever. npz quarantined like
    # the 07-28 fossils.
    "dvr-rtl-1781444700",          # Die Beet-Brüder
    # 2026-07-31 both-cold triage: two more dead merged-label fossils,
    # both heads <0.40 for days, no source anywhere to verify.
    # 989a0bea has NO channel slug (tvh-era) and a 0-120s stub label.
    "989a0bea63b249d1a6243d5f3f27e0ed",  # SpongeBob (tvh-era)
    "dvr-rtl-1780224000",          # Die Beet-Brüder (2. Fossil)
    # 2026-07-15 FP-concentration analysis on the deployed MLP3:
    # these two DEAD which=merged recordings alone carried 50% of
    # ALL measured test-frame errors (4210 of 8385) — the model
    # "false-positives" on 47-65% of their runtime, against a GT
    # that marks only ~30% ad, has ZERO confirmed_show points, and
    # can never be re-verified (no source/VOD/cache anywhere). A
    # 95.6%-acc model doesn't fail at 65% on one recording; the
    # frozen old auto-era cutlist is what's wrong (same class as
    # Kreta.de above, but which=merged so the dead-machine-label
    # retirement rule spared them).
    "dvr-kabel-eins-1779980100",   # Abenteuer Leben täglich
    "dvr-rtlzwei-1780226100",      # Von Hecke zu Hecke
    # 2026-07-20 Form()-sweep triage: these six scored IoU 0.000
    # across ALL 217 grid combos (features↔signals durations verified
    # aligned, so not a cache artifact) — the GT itself is the outlier,
    # not the params. No user labels, no archive npz. Four carry an
    # empty machine cutlist on 63-82-min recordings (n_blocks=0-as-
    # labels class), two a single tail block running exactly to the
    # recording end (captured-neighbour signature).
    "dvr-prosieben-1783011902",    # 72min, zero-block GT
    "dvr-prosieben-1783271066",    # 82min, zero-block GT
    "dvr-vox-1783008000",          # 63min, zero-block GT
    "dvr-vox-1783357200",          # 78min, zero-block GT
    "dvr-prosieben-1780544100",    # Call Me Kat — GT=[1320,1920.56] to EOF
    "dvr-prosieben-1781406105",    # Die Goldbergs — GT=[1291,1592] to EOF
    # 2026-07-22: Let's Dance 05-22 — dead rec whose cached source is
    # truncated (12570s) vs its frozen features (14056s), so the
    # signals cache is permanently tombstoned and eval falls back to
    # the naive threshold path — which ignores the per-show nn-heavy
    # override this logo-hiding show REQUIRES. Scores IoU/F1 0.00
    # forever (production would cut it fine), and its title classifies
    # as "movie", so it alone floored OVERALL(movies) to 0.32. No user
    # labels; can never be re-verified.
    "dvr-rtl-1779473700",          # Let's Dance — truncated source, naive-fallback-only
    # 2026-09-07, Simons Entscheidung nach vorgelegter Messung: ALLE
    # fuenf Let's-Dance-Aufnahmen (19.7 h, 2.4 % des Korpus) sind aus
    # dem Archiv nach tvd-train-archive-quarantine/ verschoben. Grund:
    # die Sendung dominiert jede verbliebene Fehlerkategorie des
    # Fehlerbudgets — 7 der 26 langen Fehlerlaeufe (Split-Screen-
    # Werbung), 36 % der Verwechslungsfehler der Backbone-Sonde, beide
    # langen Fehlalarme (dunkle Buehne, verstecktes Logo).
    #
    # ⚠️ DER PREIS IST GEMESSEN UND WURDE VORHER GENANNT. Trainiert mit
    # und ohne die Let's-Dance-Zeilen, bewertet auf allem, was NICHT
    # Let's Dance ist, fuenf gepaarte Seeds: F1 0.9069 -> 0.8982,
    # Median-Delta -0.0087, positiv in 1 von 5. Bei sd 0.0045 rund zwei
    # Standardabweichungen. Die Sendung ist schwer, aber sie LEHRT:
    # dunkle Buehne, verstecktes Logo und Split-Screen kommen auch
    # anderswo vor, nur seltener. Wer diese Zeilen je wieder
    # hereinholt, macht das Modell auf allem anderen besser.
    #
    # Rueckgaengig: die npz aus tvd-train-archive-quarantine/ zurueck
    # nach tvd-train-archive/ und diese vier Zeilen entfernen.
    "dvr-rtl-1780078500",          # Let's Dance — 2026-09-07 quarantaeniert
    "dvr-rtl-1780683300",          # Let's Dance — 2026-09-07 quarantaeniert
    "a307345d5a95af506072a426d1bf80ea",  # Let's Dance — 2026-09-07 quarantaeniert
    "e08d22bae601f4ef6e79da3691fafffb",  # Let's Dance — 2026-09-07 quarantaeniert
    # 2026-07-24 label audit (prosieben-1779878100 was 0.00/0.00 on BOTH
    # heads in the both-heads-cold report): a whole BATCH of archive npz
    # written 06-02 07:41 — i.e. BEFORE the 06-04 source-cache truncation
    # guard ([[source_cache_truncation_silent]]) — were extracted from a
    # source truncated at ~40-45min. Signature (swept across all archives):
    # the meta "ads" list carries a block whose END is BEYOND the label
    # horizon while labels are all-zero (ad_frac 0.000).
    #
    # These entries are now BELT-AND-SUSPENDERS: the 11 npz themselves were
    # QUARANTINED out of the corpus (~/.cache/tvd-train-archive-quarantine/)
    # because — contrary to the first read of this audit — the all-zero
    # labels are POISON in TRAIN, not merely incomplete negatives. Proof:
    # moving prosieben-1779878100 test→train in the 07-24 15:37 run (via an
    # earlier TEST_SET_EXCLUDE-only fix that kept them train-eligible)
    # crashed the sibling reviewed BBT dvr-prosieben-1782482508 from a stable
    # 0.88 (3 prior runs) to 0.45 — the deterministic MLP (random_state=0)
    # learned "BBT = no ad" from the mislabelled rec. A truncated sitcom with
    # its ad-break cut off and labelled all-show is an unrepresentative
    # negative that biases the whole show's prior. Unfixable (no source to
    # re-detect), so they're removed from BOTH roles. If a quarantined npz is
    # ever restored, these keep it out of TEST too. 11 recs, 2.5-Men / Big
    # Bang / Charmed midday reruns. (Galileo dvr-prosieben-1780506300 hit the
    # truncation signature too but is a real 65min rec, ad_frac 0.311, minor
    # tail overrun — NOT truncated, left in the corpus.)
    "dvr-prosieben-1779870300",    # Two and a Half Men — trunc 40min, quarantined
    "dvr-prosieben-1779871800",    # Two and a Half Men — trunc 45min, quarantined
    "dvr-prosieben-1779873600",    # Two and a Half Men — trunc 40min, quarantined
    "dvr-prosieben-1779875100",    # The Big Bang Theory — trunc 40min, quarantined
    "dvr-prosieben-1779878100",    # The Big Bang Theory — trunc 40min, quarantined (crashed sibling)
    "dvr-prosieben-1779884400",    # Two and a Half Men — trunc 40min, quarantined
    "dvr-prosieben-1779885900",    # Two and a Half Men — trunc 45min, quarantined
    "dvr-prosieben-1779889301",    # The Big Bang Theory — trunc 42min, quarantined
    "dvr-sixx-1779894000",         # Charmed — trunc 40min, quarantined
    "dvr-sixx-1779980700",         # Charmed — trunc 40min, quarantined
    "dvr-sixx-1780069500",         # Charmed — trunc 45min, quarantined
    # 2026-07-27 corpus-wide label audit (scripts/corpus-label-audit.py):
    # 16 of 486 archived recordings carry labels that contradict their OWN
    # per-second NN signal — the head is confidently "ad" for minutes in a
    # stretch the labels call show, or the reverse. Same poison class as the
    # 07-24 batch above, found systematically rather than by chance: the
    # audit recomputes the deployed head over the archive's own features, so
    # it needs no decode and covers every recording, not just the ones with
    # a signals dump. Verified before use — its verdict matched the real
    # dump on 35 of 35 recordings that had both.
    #
    # These seven are DEAD (no source, no VOD) and machine-labelled, so the
    # labels can never be re-derived. Their npz are in
    # ~/.cache/tvd-train-archive-quarantine/ ; these entries keep them out
    # of TEST too should an npz ever be restored. Two of them carry ZERO ad
    # blocks against 1254 s and 785 s of confident ad signal — the
    # n_blocks=0-as-labels class that provably crashed a sibling recording
    # from 0.88 to 0.45 on 07-24.
    "dvr-sixx-1780040100",         # Charmed — 0 blocks vs 1254s ad signal
    "dvr-sixx-1779954000",         # Charmed — 0 blocks vs 785s ad signal
    "dvr-rtl-1780832400",          # Die Beet-Brüder — 826s ad outside labels
    "dvr-sixx-1779951600",         # Charmed — 493s ad outside labels
    "dvr-nick-1781033400",         # Futurama — 411s ad outside labels
    "dvr-nick-1781031900",         # Futurama — 382s ad outside labels
    "dvr-sixx-1780291500",         # Charmed — 374s ad outside labels
    # 2026-07-28, second pass: these three carry HUMAN labels in the
    # archive but none in the live system — ads.json is empty for all
    # three, two are gone from the DVR grid entirely, and their npz are
    # frozen at 25 June / 18 July because no npz is written for a
    # label-less recording. So the archive is the ONLY place these labels
    # still exist, and the head contradicts them over 250-460 s. Removing
    # them is not overruling a reviewer: the reviewer's verdict is already
    # gone everywhere else. dvr-kabel-eins-1779980100 — the same show, the
    # same signature — was quarantined for exactly this on 07-15.
    "dvr-vox-1781036100",          # Hot oder Schrott — 463s hole, dead
    "dvr-kabel-eins-1780325700",   # Abenteuer Leben täglich — 391s hole, dead
    "dvr-nick-1778954400",         # SpongeBob — 257s phantom, live ads empty
}


def quarantaene(archiv):
    """uuids, deren Archiv-Eintrag nach <archiv>-quarantine/ verschoben wurde."""
    q = Path(archiv).with_name(Path(archiv).name + "-quarantine")
    return {p.stem for p in q.glob("*.npz")} if q.is_dir() else set()


def ausgeschlossen(archiv):
    """Alles, was nie als Messziel dienen darf: Liste UND Quarantaene."""
    return set(TEST_SET_EXCLUDE) | quarantaene(archiv)
