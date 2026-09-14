#!/bin/bash
# Eine registrierte Frage an EINEM Nachmittag beantworten: je Arm ein
# Prozess (der M5 Pro hat 18 Kerne, ein Fit nutzt einen), Paarung ueber
# gemeinsamen Zeitstempel + gemeinsame Seeds, am Ende rechnet das Audit.
#
#   tv-tagesserie.sh "armMit,armOhne" [N=5] [flagsMit] [flagsOhne]
#
# Die beiden letzten Argumente sind zusaetzliche Schalter je Arm, z.B.
#   tv-tagesserie.sh "mlp32-cwtmpwm-streng,mlp32-cwtmpwm-belegt" 5 "--herkunft-streng"
# Eine Frage, die kein Spalten- sondern ein Gewichtungs-Unterschied ist,
# kann NUR so gefahren werden: der Schalter wirkt beim Aufbau des Korpus,
# also vor dem Armlauf, und zwei Arme in einem Prozess messen zweimal
# dasselbe.
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
ZUSATZ_MIT="${3:-}"
ZUSATZ_OHNE="${4:-}"

PY="$HOME/ml/tv-classifier/.venv/bin/python"
REPO="$HOME/src/tv-detect"
ECHT="$HOME/.cache/tvd-train-archive"
LEHRER="$HOME/.cache/tv-train-head-out"   # deployter Champion = Hygiene-Lehrer
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
# ⚠️ EIN Stichtag fuer BEIDE Arm-Prozesse. Ohne ihn liest jeder Prozess
# beim Bauen der Gewichte seine eigene Uhr; age_mult faellt dann pro Lauf
# anders aus, und zwei Prozesse mit identischer Konfiguration und gleichem
# Seed weichen um Median 0.0073 im golden_median ab (2026-09-06, Ledger
# §3ar) — mehr als jeder Effekt, den dieses Skript je gemessen hat. Auf
# die volle Stunde gerundet, dieselbe Konvention wie im Nightly.
STICHTAG=$(( $(date +%s) / 3600 * 3600 ))
echo "Tagesserie $TS — Arme: $ARME, $N Paare, Seeds: $SEEDS, Stichtag: $STICHTAG"

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
  # ⚠️ BEIDE Arme bekommen DENSELBEN eingefrorenen Label-Hygiene-Lehrer.
  # Der Lehrer wird aus dem --output-PFAD geladen, und am Ende seines
  # Laufs schreibt ein Arm genau dorthin seinen Kopf. Teilen sich zwei
  # Arme den Pfad, lernt Arm 2 mit dem Modell, das Arm 1 gerade erzeugt
  # hat -- gemessen am 2026-09-13: Arm 1 lud einen 1282-Spalten-Lehrer,
  # Arm 2 einen mit 1301 Spalten, und beide verwarfen unterschiedlich
  # viele Frames. Der Vergleich mass dann zwei Dinge gleichzeitig.
  # Der Champion ist zugleich der Lehrer der Produktion -- ohne ihn
  # liefe die Serie ganz ohne Hygiene und damit neben dem Nightly her.
  if [ -f "$LEHRER/head.bin" ]; then
    cp "$LEHRER/head.bin" "$D/out/head.bin"
    for _s in "$LEHRER"/head.*.json; do
      [ -f "$_s" ] && cp "$_s" "$D/out/$(basename "$_s")"
    done
  else
    echo "  ⚠ kein Champion unter $LEHRER — die Serie laeuft OHNE "
    echo "    Label-Hygiene und damit nicht auf Produktions-Stand."
  fi
done

RC=0
for ARM in "${ARMLISTE[@]}"; do
  D="$BASIS/$ARM"
  if [ "$ARM" = "${ARMLISTE[0]}" ]; then ZUSATZ="$ZUSATZ_MIT"; else ZUSATZ="$ZUSATZ_OHNE"; fi
  # bash 3.2 + set -u: ein leeres Array darf nicht nackt expandiert werden.
  ZUSATZ_ARR=()
  [ -n "$ZUSATZ" ] && read -ra ZUSATZ_ARR <<< "$ZUSATZ"
  # ⚠️ LAUT sagen, welcher Arm was bekommen hat. Eine stille Asymmetrie
  # zwischen den Armen ist genau der Fehler, den diese Serie messen soll,
  # und nicht der, den sie machen darf.
  # ⚠️ NIE einen Kommentar zwischen zwei fortgesetzte Zeilen eines
  # Aufrufs setzen. Der Backslash klebt die Zeilen zusammen, das `#`
  # beginnt dann mitten im Befehl einen Kommentar und verschluckt ALLE
  # folgenden Schalter — `bash -n` findet das nicht, der Lauf startet mit
  # halber Konfiguration. Am 2026-09-14 beim Haerten dieses Skripts
  # gebaut und in einer Probe gefangen, bevor es lief.
  #
  # --audio-dynamik und --herkunft-belegt stehen unten fuer
  # PRODUKTIONS-PARITAET: beide sind im Nightly scharf (O22 seit 09-08,
  # O17 davor). Eine Serie ohne sie beantwortet ihre Frage an einem
  # Modell, das es nicht mehr gibt.
  #
  # --sealed-frac bleibt BEWUSST weg (Nightly: 0.20). Versiegelt wird nur,
  # was noch nicht im Ledger steht; bei 0.0 erben beide Arme denselben
  # Stand aus der Archiv-Kopie und bleiben symmetrisch. Mit 0.20 wuerde
  # jeder Arm neue Aufnahmen fuer sich versiegeln — die Arme trennten
  # sich im Korpus, und genau das soll hier nicht passieren. Preis: der
  # Trainingssatz der Serie waechst langsam ueber den der Produktion
  # hinaus.
  echo "  Arm $ARM: Log $D/lauf.log, Zusatz-Schalter: ${ZUSATZ:-keine}"
  "$PY" "$REPO/scripts/train-head.py" \
      --workers 4 \
      --backbone "$HOME/.cache/tv-detect-daemon/backbone.onnx" \
      --logo-dir "$HOME/.cache/tv-detect-daemon/logos" \
      --output "$D/out/head.bin" \
      --hls-root /tmp/tv-train-snapshot \
      --with-logo --with-audio --with-minute-prior --with-self-training \
      --audio-dynamik \
      --herkunft-belegt \
      --train-archive "$D/archive" \
      --serie-archiv "$ECHT" \
      --head-arch mlp32-channel-whisper-temporal-mp-wm \
      --shadow-eval --nur-cache --nur-tagesserie \
      --tagesserie "$N" \
      --tagesserie-arme "$ARME" \
      --tagesserie-nur-arm "$ARM" \
      --tagesserie-ts "$TS" \
      --tagesserie-seeds "$SEEDS" \
      --stichtag "$STICHTAG" \
      ${ZUSATZ_ARR[@]+"${ZUSATZ_ARR[@]}"} \
      >"$D/lauf.log" 2>&1 || RC=1
done
echo ""
echo "=== Laeufe beendet (rc=$RC) — das Urteil rechnet das Audit: ==="
"$PY" "$REPO/scripts/audit-preregistration.py"
