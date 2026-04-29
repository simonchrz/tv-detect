#!/bin/bash
# Daily summary of the tv-detect autonomous learning loop. Writes a
# plain-text report to disk + best-effort emails it. Triggered by
# launchd at 08:00 daily — that's after the 03:30 retrain so the
# numbers are fresh, and before the user typically uses the system.
#
# Thin wrapper around the gateway's /api/learning/summary endpoint
# (= single JSON with all metrics) + /api/learning/fingerprint-scan
# (= POST-trigger that auto-confirms recordings against show
# fingerprints and reports counts). Replaces the prior in-script
# Python that read SMB files directly — same TCC-bypass benefit as
# the rec/live daemons, plus the gateway already has the helpers.
#
# Email path uses macOS's /usr/bin/mail (postfix) — works if
# /etc/postfix is configured for outbound, otherwise the report
# just lands on disk. Either way, never fails the launchd job.

LOG="$HOME/Library/Logs/tv-detect-daily.log"
SUMMARY="$HOME/Library/Logs/tv-detect-daily-summary.txt"
EMAIL="simonch@gmx.de"
GATEWAY_HTTPS="https://raspberrypi5lan:8443"
exec >>"$LOG" 2>&1
echo "=== $(date '+%F %T') ==="

/usr/bin/python3 - >"$SUMMARY" 2>&1 <<PY
import json, ssl, time, urllib.request

ctx = ssl.create_default_context()
ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE

def http_get(path):
    with urllib.request.urlopen(f"$GATEWAY_HTTPS{path}",
                                  timeout=30, context=ctx) as r:
        return json.loads(r.read())

def http_post(path):
    req = urllib.request.Request(f"$GATEWAY_HTTPS{path}", method="POST")
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        return json.loads(r.read())

s = http_get("/api/learning/summary")
fp = http_post("/api/learning/fingerprint-scan")

m = s["model"]; r = s["recordings"]; lq = s["labelling_queue"]
st = s["self_training"]; ch = s["channel_health"]

L = []
L.append(f"tv-detect tägliche Zusammenfassung — {time.strftime('%a %F')}")
L.append("=" * 60)
L.append("")
L.append("MODELL")
L.append(f"  Test IoU:       {m['test_iou']:.2f}" if m['test_iou'] else "  Test IoU: —")
L.append(f"  Test Acc:       {m['test_acc']*100:.1f} %" if m['test_acc'] else "  Test Acc: —")
if m["drift_vs_7d_median"] is not None:
    d = m["drift_vs_7d_median"]
    sym = "📉" if d < -0.03 else ("📈" if d > 0.03 else "→")
    L.append(f"  Drift vs 7-d Median: {d:+.2f} {sym}")
L.append(f"  Runs letzte 24 h:  {m['runs_24h']} ({m['deployed_24h']} deployed, "
         f"{m['rejected_24h']} rejected)")
L.append("")
L.append("AUFNAHMEN")
L.append(f"  Gesamt:                {r['total']}")
L.append(f"  Neu in 24 h:           {r['new_24h']}")
L.append(f"  Mit User-Edits:        {r['with_user']}")
L.append(f"  Davon ✓ geprüft:       {r['reviewed']}")
L.append(f"  User-Edits in 24 h:    {r['user_edits_24h']}")
L.append(f"  Skip-Bestätigungen:    {r['skip_signals']} insgesamt")
L.append("")
L.append("LABELLING-QUEUE")
L.append(f"  Aktive Targets offen:  {lq['open']}")
L.append(f"  Geprüft (ausgeblendet):{lq['reviewed_skipped']}")
L.append("")
L.append("SELF-TRAINING")
L.append(f"  Aufnahmen mit Pseudo-Labels: {st['recordings_with_pseudo']}")
L.append(f"  Pseudo-Frames insgesamt:     {st['frames_total']}  "
         f"(ad={st['frames_ad']}  show={st['frames_show']})")
L.append("")
L.append("AUTO-CONFIRM (Show-Fingerprint)")
L.append(f"  Aktive Fingerprints:   {fp.get('fingerprints_used', 0)}")
L.append(f"  Heute auto-bestätigt:  {fp.get('matched', 0)}")
L.append(f"  Mismatch (zur Review): {len(fp.get('skipped', []))}")
L.append("")
if ch.get("broken"):
    L.append("⚠ AUFFÄLLIGKEITEN")
    for slug in ch["broken"]:
        L.append(f"  {slug}: letzte 5 Aufnahmen ohne Block — Logo-Template kaputt?")
    L.append("")
L.append("Details: $GATEWAY_HTTPS/learning")
print("\n".join(L))
PY

# macOS notification via osascript — replaces the never-arriving
# postfix mail. The notification banner shows the most-relevant
# headline numbers (Test IoU, Drift, neue Aufnahmen, offene Targets).
# Full plain-text report stays at $SUMMARY for follow-up. Click on
# the banner does nothing (no action handler) — open /learning
# manually if you want details.
if [ -s "$SUMMARY" ]; then
  TITLE="tv-detect Tagesbericht $(date +%F)"
  BODY=$(grep -E "Test IoU|Neu in 24 h|Aktive Targets|Drift vs" "$SUMMARY" \
         | sed 's/  */ /g; s/^ *//' \
         | head -4 \
         | sed 's/"/\\"/g' \
         | awk 'BEGIN{ORS="\\n"} {print}')
  /usr/bin/osascript -e \
    "display notification \"$BODY\" with title \"$TITLE\" subtitle \"Details: /learning\""
  echo "notification posted (full report: $SUMMARY)"
fi
