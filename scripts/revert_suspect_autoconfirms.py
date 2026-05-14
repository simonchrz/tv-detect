#!/usr/bin/env python3
"""Revert auto-confirmed-empty ads_user.json files in cohorts where
other recordings have user-confirmed ad blocks (= SUSPECT cohort,
auto-confirm likely false-positive due to under-bumper-coverage).

For each suspect entry:
  1. Delete ads_user.json (= clears the false 'reviewed' signal)
  2. Write .detect-requested marker (= daemon re-detects with the
     new bumper templates + current head)

Run on Pi. Dry-run by default; pass --apply to actually do it."""
import os, sys, json, urllib.request, time
from pathlib import Path
from collections import defaultdict

GATEWAY = os.environ.get("GATEWAY", "http://localhost:8080")
TVH_BASE = "http://localhost:9981"
HLS_DIR = Path("/mnt/tv/hls")


def main():
    apply = "--apply" in sys.argv
    data = json.loads(urllib.request.urlopen(
        f"{TVH_BASE}/api/dvr/entry/grid?limit=2000", timeout=10).read())
    by_cohort = defaultdict(lambda: {"user_ads": [], "auto_empty": []})
    for e in data.get("entries", []):
        if e.get("sched_status") not in ("completed", "completedError"): continue
        uuid = e.get("uuid")
        if not uuid: continue
        title = (e.get("disp_title") or "").strip()
        ch = (e.get("channelname") or "").strip()
        if not title or not ch: continue
        user_p = HLS_DIR / f"_rec_{uuid}" / "ads_user.json"
        if not user_p.is_file(): continue
        try:
            d = json.loads(user_p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict): continue
        ads = d.get("ads") or []
        cohort = (title, ch)
        if ads:
            by_cohort[cohort]["user_ads"].append(uuid)
        elif d.get("auto_confirmed_at"):
            by_cohort[cohort]["auto_empty"].append(uuid)

    suspect_cohorts = {c for c, v in by_cohort.items()
                       if v["user_ads"] and v["auto_empty"]}
    print(f"Suspect cohorts (have BOTH user-ads AND auto-empty): "
          f"{len(suspect_cohorts)}")
    n_revert = 0
    for c in sorted(suspect_cohorts):
        v = by_cohort[c]
        title, ch = c
        n = len(v["auto_empty"])
        n_revert += n
        print(f"  {title[:35]:35} {ch[:18]:18} -> revert {n} "
              f"(GT: {len(v['user_ads'])} user-confirmed)")

    print(f"\nTotal to revert: {n_revert}")
    if not apply:
        print("(dry-run; pass --apply to execute)")
        return

    n_done = 0; n_err = 0
    now = int(time.time())
    for c in suspect_cohorts:
        for uuid in by_cohort[c]["auto_empty"]:
            rec = HLS_DIR / f"_rec_{uuid}"
            try:
                (rec / "ads_user.json").unlink()
                # Drop ads.json too so /ads endpoint regenerates from
                # cutlist (= will be empty, but the daemon will re-
                # detect on next cycle and refill it).
                ads_p = rec / "ads.json"
                if ads_p.exists(): ads_p.unlink()
                (rec / ".detect-requested").write_text(
                    json.dumps({"ts": now}))
                n_done += 1
            except Exception as ex:
                n_err += 1
                print(f"  ERR {uuid[:8]}: {ex}")
    print(f"\nReverted: {n_done}  errors: {n_err}")
    print("Daemon will re-detect on next cycle (~2-3 min). With new "
          "bumper templates + current head, suspect cohorts should "
          "now find their actual ads.")


if __name__ == "__main__":
    main()
