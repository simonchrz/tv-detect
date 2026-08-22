#!/bin/bash
# Eine registrierte Frage an EINEM Nachmittag beantworten: je Arm ein
# Prozess (der M5 Pro hat 18 Kerne, ein Fit nutzt einen), Paarung ueber
# gemeinsamen Zeitstempel + gemeinsame Seeds, am Ende rechnet das Audit.
#
#   tv-tagesserie.sh "armMit,armOhne" [N=5]
#
# ⚠️ Was hier absichtlich NICHT passiert:
#   * Kein Urteil. Das Skript druckt am Ende das Audit — die Regel
#     entscheidet, nicht der Lauf und nicht der Aufrufer.
#   * Kein Schreiben ins echte Trainings-Archiv. Jeder Arm-Prozess laeuft
#     gegen eine eigene KOPIE (Split-Ledger, .npz); nur die Serien-Zeilen
#     gehen ueber --serie-archiv ins echte shadow-trend.jsonl.
#   * Kein TVD_LAUF=nightly. Die Zeilen tragen quelle=tagesserie und
#     zaehlen ausschliesslich fuer tagesserie-registrierte Regeln.
#
# ⚠️ ABGEKOPPELT starten (nohup ... & mit eigenem Log), NIE im Vordergrund
# eines Werkzeugs mit Timeout: die Laeufe vom 20. und 21.08. starben nach
# ~1 min still mit dem Prozessbaum des Aufrufers — kein Fehler im lauf.log,
# Arm 2 lief nie an, das Audit meldete nur "Serie hat noch nicht begonnen".
set -u
ARME="${1:?Arme fehlen, z.B. 'mlp32-channel-whisper-temporal-mp-wm,mlp32'}"
N="${2:-5}"

PY="$HOME/ml/tv-classifier/.venv/bin/python"
REPO="$HOME/src/tv-detect"
ECHT="$HOME/.cache/tvd-train-archive"
TS="$(date +%Y%m%dT%H%M%S)"
BASIS="${TMPDIR:-/tmp}/tv-tagesserie-$TS"
mkdir -p "$BASIS"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"

# Seeds hier wuerfeln und BEIDEN Prozessen mitgeben — sonst paart das
# Audit nichts.
SEEDS=$("$PY" -c "
import hashlib
b = int(hashlib.sha256('$TS'.encode()).hexdigest()[:8], 16) % 10000
print(','.join(str((b + 1 + 997*i) % 10000) for i in range($N)))")
echo "Tagesserie $TS — Arme: $ARME, $N Paare, Seeds: $SEEDS"

"$PY" "$HOME/bin/tv-train-snapshot-fetch.py" \
  --gateway-url https://raspberrypi5lan:8443 --out /tmp/tv-train-snapshot \
  || { echo "Snapshot-Fetch scheiterte"; exit 1; }

# ⚠️ Arme NACHEINANDER, nicht parallel. Der erste Parallel-Versuch
# (2026-08-12) endete mit Killed:9 durch den OOM-Killer: jeder Arm baut
# eine ~10-GB-Trainingsmatrix, zwei gleichzeitig drueckten die Maschine
# in 50 GB Swap. Sequentiell kostet die Fits-Phase doppelt — dafuer
# stirbt nichts still nach 40 Minuten Vorlauf. Wer Parallelitaet will,
# misst vorher den freien Speicher, nicht hinterher den Swap.
# Beide Archiv-Kopien VOR dem ersten Fit ziehen — Kopien vom selben
# Stand, sonst misst der zweite Arm einen anderen Korpus (die zweite
# Lehre aus dem 2026-08-12-Versuch).
IFS=',' read -ra ARMLISTE <<< "$ARME"
for ARM in "${ARMLISTE[@]}"; do
  D="$BASIS/$ARM"
  mkdir -p "$D/out"
  cp -R "$ECHT" "$D/archive"
done

RC=0
for ARM in "${ARMLISTE[@]}"; do
  D="$BASIS/$ARM"
  echo "  Arm $ARM: Log $D/lauf.log"
  "$PY" "$REPO/scripts/train-head.py" \
      --workers 4 \
      --backbone "$HOME/.cache/tv-detect-daemon/backbone.onnx" \
      --logo-dir "$HOME/.cache/tv-detect-daemon/logos" \
      --output "$D/out/head.bin" \
      --hls-root /tmp/tv-train-snapshot \
      --with-logo --with-audio --with-minute-prior --with-self-training \
      --train-archive "$D/archive" \
      --serie-archiv "$ECHT" \
      --head-arch mlp32-channel-whisper-temporal-mp-wm \
      --shadow-eval --nur-cache --nur-tagesserie \
      --tagesserie "$N" \
      --tagesserie-arme "$ARME" \
      --tagesserie-nur-arm "$ARM" \
      --tagesserie-ts "$TS" \
      --tagesserie-seeds "$SEEDS" \
      >"$D/lauf.log" 2>&1 || RC=1
done
echo ""
echo "=== Laeufe beendet (rc=$RC) — das Urteil rechnet das Audit: ==="
"$PY" "$REPO/scripts/audit-preregistration.py"
