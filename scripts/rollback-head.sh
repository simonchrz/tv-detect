#!/bin/bash
# Roll the deployed ad-detection head back to an archived nightly champion.
#
# Since 2026-05-30 the trainer archives the FULL bundle per deploy
# (head.<ts>.bin + its channel-map / calibration / test-set sidecars) under
# $TRAIN_OUT/archive/. This restores one such bundle to the Pi gateway via
# the same /api/internal/head-bundle endpoint the trainer uses (atomic
# extract on the Pi, head.bin written last). The gateway keeps the replaced
# bundle as a rollback-bak.
#
#   rollback-head.sh list             # restorable bundles (ts, slugs, IoU)
#   rollback-head.sh restore <ts>     # restore head.<ts>.* to the Pi
#
# REFUSES to restore a head WITHOUT its matching channel-map sidecar:
# restoring a channel-aware head with the wrong/missing channel-map
# misaligns the channel one-hot columns -> degraded/broken inference. This
# is exactly the trap that made the 2026-05-30 regression un-rollbackable
# (head.bin was archived but the champion's 10-slug channel-map was gone).
# Pre-2026-05-30 archives are head-only and therefore NOT safely restorable.
set -euo pipefail

# ⚠️ MUSS zum Ziel des Trainers passen (tv-train-head.sh: GATEWAY=
# https://raspberrypi5lan:8443). Hier stand bis 2026-08-05
# http://raspberrypi5lan:8080 — der hls-gateway, der am 2026-06-01
# stillgelegt wurde. Ein Rollback wäre also genau dann gescheitert, wenn
# man ihn braucht: nach einem schlechten Deploy, unter Zeitdruck.
GATEWAY="${GATEWAY:-https://raspberrypi5lan:8443}"
CURL_OPTS=(-sk)  # -k: Caddy fährt das interne Zertifikat
ARCHIVE="${TRAIN_OUT:-$HOME/.cache/tv-train-head-out}/archive"
HISTORY="${TRAIN_OUT:-$HOME/.cache/tv-train-head-out}/head.history.json"

usage() { echo "usage: $0 {list | restore <ts>}" >&2; exit 1; }
[ $# -ge 1 ] || usage

iou_for_ts() {  # best-effort median-IoU lookup from history.json
  [ -f "$HISTORY" ] || { echo "?"; return; }
  python3 - "$1" "$HISTORY" <<'PY' 2>/dev/null || echo "?"
import json,sys
ts,hp=sys.argv[1],sys.argv[2]
h=json.load(open(hp))
es=h if isinstance(h,list) else h.get("entries",h.get("history",[]))
for e in es:
    if e.get("ts")==ts:
        v=e.get("test_iou_tv_median",e.get("test_iou_median",e.get("test_iou")))
        print(f"{v:.3f}" if isinstance(v,(int,float)) else "?"); break
else: print("?")
PY
}

case "$1" in
  list)
    [ -d "$ARCHIVE" ] || { echo "no archive at $ARCHIVE"; exit 1; }
    echo "restorable bundles in $ARCHIVE:"
    found=0
    for h in "$ARCHIVE"/head.*.bin; do
      [ -e "$h" ] || continue
      found=1
      b=$(basename "$h" .bin); ts=${b#head.}
      cm="$ARCHIVE/head.$ts.channel-map.json"
      sz=$(wc -c < "$h" | tr -d ' ')
      iou=$(iou_for_ts "$ts")
      if [ -f "$cm" ]; then
        n=$(grep -oE '"n":[ ]*[0-9]+' "$cm" | grep -oE '[0-9]+' | head -1)
        printf "  %-18s %7s B  IoU %-6s  channel-map %s slugs  [full ✓]\n" "$ts" "$sz" "$iou" "$n"
      else
        printf "  %-18s %7s B  IoU %-6s  [head-only — NOT restorable]\n" "$ts" "$sz" "$iou"
      fi
    done
    [ "$found" = 1 ] || echo "  (none)"
    # ⚠️ Diese Liste ist NICHT das ganze Sicherheitsnetz. Sie zeigt nur den
    # lokalen Bestand; vollständige Bündel liegen zusätzlich off-site als
    # GitHub-Releases (model-anchor.sh, ein Anker je echtem Deploy, mit
    # Kanal-Karte). Am 2026-08-05 hat mich genau diese Verkürzung in die
    # Irre geführt: die lokale Liste zeigte drei Einträge, ich schloss
    # daraus „Champion unwiederbringlich" — und off-site lagen vierzehn,
    # der gesuchte darunter. Deshalb steht der Hinweis hier und nicht in
    # einer README, die im Ernstfall niemand liest.
    echo
    echo "  weiter zurück? Off-Site-Anker (mit Kanal-Karte):"
    echo "    ./scripts/model-anchor.sh list"
    echo "    ./scripts/model-anchor.sh install model-anchor-auto-<ts>"
    ;;

  restore)
    [ $# -eq 2 ] || usage
    ts="$2"
    h="$ARCHIVE/head.$ts.bin"
    cm="$ARCHIVE/head.$ts.channel-map.json"
    [ -f "$h" ] || { echo "no archived head for ts=$ts (try: $0 list)" >&2; exit 1; }
    [ -f "$cm" ] || { echo "REFUSING: head.$ts has no channel-map sidecar — restoring it would misalign channel one-hots (degraded inference). Head-only archive, not safely restorable." >&2; exit 1; }

    stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT
    cp "$h" "$stage/head.bin"
    for suf in channel-map calibration test-set minute-prior; do
      [ -f "$ARCHIVE/head.$ts.$suf.json" ] && cp "$ARCHIVE/head.$ts.$suf.json" "$stage/head.$suf.json"
    done
    # ⚠️ Ein MLP4-Kopf braucht seine eigene minute-prior-Tabelle; das Archiv
    # legt sie bis heute nicht ab (dort liegen head.<ts>.bin + channel-map/
    # calibration/test-set). Der zurueckgerollte Kopf rechnet dann gegen die
    # Tabelle der Gegenwart. NICHT abbrechen wie beim fehlenden channel-map:
    # dort waeren die One-Hots verschoben (kaputte Inferenz), hier sind es
    # gemessen -0.004 IoU (2026-08-05, 12 Aufnahmen). Sichtbar muss es aber
    # sein — genau so ist es am 05-08 unbemerkt passiert.
    if [ "$(head -c4 "$stage/head.bin")" = "MLP4" ] && \
       [ ! -f "$stage/head.minute-prior.json" ]; then
      echo "  WARNUNG: head.$ts ist MLP4, das Archiv hat keine passende" >&2
      echo "  minute-prior.json — der Kopf laeuft gegen die aktuelle Tabelle" >&2
      echo "  (~-0.004 IoU). Der naechste Nightly-Deploy paart wieder sauber." >&2
    fi
    # ⚠️ head.gate.bin gehoert zum ERSETZTEN Kopf, nicht zu diesem. Bliebe es
    # liegen, verglichen die naechsten Naechte den Kandidaten gegen einen
    # Kopf, der gar nicht mehr deployt ist — das Gate verteidigte ein
    # Qualitaetsniveau, das die Produktion nicht hat, und koennte einen
    # Kandidaten ablehnen, der den TATSAECHLICH deployten schlaegt.
    # train-head.py faellt ohne gate.bin auf head.bin zurueck, also auf genau
    # den zurueckgerollten Kopf. Beobachtet am 05-08: head.bin 07-29,
    # head.gate.bin 08-04.
    # Das Gateway kennt nur head-bundle (schreiben), kein Loeschen — also
    # per ssh. Schlaegt das fehl (Fernzugriff ohne ssh), muss es sichtbar
    # gesagt werden statt still zu unterbleiben.
    GATE_REMOTE="${PI_HOST:-raspberrypi5lan}"
    echo "  raeume head.gate.bin (gehoert zum ersetzten Kopf) ..."
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$GATE_REMOTE" \
         "rm -f /mnt/tv/hls/.tvd-models/head.gate.bin" 2>/dev/null; then
      echo "  ACHTUNG: head.gate.bin konnte nicht entfernt werden." >&2
      echo "  Bis das passiert, vergleicht der naechtliche Gate-Test gegen den" >&2
      echo "  ERSETZTEN Kopf und kann einen Kandidaten ablehnen, der den jetzt" >&2
      echo "  deployten schlaegt. Von Hand:" >&2
      echo "    ssh $GATE_REMOTE rm /mnt/tv/hls/.tvd-models/head.gate.bin" >&2
    fi
    ( cd "$stage" && tar czf bundle.tar.gz head.bin head.*.json )

    echo "rolling back to head.$ts (IoU $(iou_for_ts "$ts")) → $GATEWAY …"
    resp=$(curl -fsS "${CURL_OPTS[@]}" -X POST --data-binary "@$stage/bundle.tar.gz" \
        -H "Content-Type: application/gzip" \
        "$GATEWAY/api/internal/head-bundle")
    echo "  gateway: $resp"
    echo "✓ rolled back to head.$ts — gateway keeps the replaced head as a rollback-bak."
    echo "  (verify on /learning; the daemon picks up head.bin via mtime watch)"
    ;;

  *) usage ;;
esac
