#!/bin/bash
# SigLIP-Herausforderer (docs/siglip-spur-design.md, Schritt 6.3 vorgezogen):
# zwei Arme auf IDENTISCHEN Daten, nacheinander (Speicher, Memory
# training_stirbt_am_speicher):
#   kontrolle  genau die Nightly-Schalter (MLP6; --head-arch mlp32 wie
#              TVH_HEAD_ARCH_OVERRIDE in tv-train-head.sh — dort gewinnt der
#              letzte Wert, also laeuft die Produktion NACKT)
#   siglip     dieselben + --siglip-spalten (MLP7)
# Muster wie tv-tagesserie.sh: je Arm eine ARCHIV-KOPIE und eine Kopie des
# Champion-Buendels (Kopf-an-Kopf + Label-Hygiene-Lehrer), ein eigener
# Snapshot, eigener Ausgabeordner. Schreibt NICHTS in ~/.cache/tvd-train-archive,
# ~/.cache/tv-train-head-out oder auf den Pi. KEIN Upload (der steckt nur in
# tv-train-head.sh). --sealed-frac bleibt weg wie in der Tagesserie, damit
# beide Arme denselben Ledger-Stand erben.
set -u
PY="$HOME/ml/tv-classifier/.venv/bin/python"
REPO="$HOME/src/tv-detect"
ECHT="$HOME/.cache/tvd-train-archive"
CHAMP="$HOME/.cache/tv-train-head-out"
TS=$(date +%Y%m%dT%H%M%S)
BASIS="$HOME/.cache/tvd-siglip-herausforderer/$TS"
STICHTAG=$(( $(date +%s) / 3600 * 3600 ))
mkdir -p "$BASIS"
cp -R /tmp/tv-train-snapshot "$BASIS/snapshot" || { echo "Snapshot-Kopie gescheitert"; exit 1; }
for ARM in kontrolle siglip; do
  D="$BASIS/$ARM"; mkdir -p "$D/out"
  cp -R "$ECHT" "$D/archive" || { echo "Archiv-Kopie $ARM gescheitert"; exit 1; }
  cp "$CHAMP/head.bin" "$CHAMP/head.gate.bin" "$D/out/" 2>/dev/null
  for _s in "$CHAMP"/head.*.json; do cp "$_s" "$D/out/"; done
done
echo "Basis $BASIS, Stichtag $STICHTAG"
RC=0
for ARM in kontrolle siglip; do
  D="$BASIS/$ARM"
  ZUSATZ=(); [ "$ARM" = siglip ] && ZUSATZ=(--siglip-spalten)
  echo "$(date +%T) Arm $ARM: Log $D/lauf.log ${ZUSATZ[@]+"${ZUSATZ[@]}"}"
  "$PY" "$REPO/scripts/train-head.py" \
      --workers 4 \
      --backbone "$HOME/.cache/tv-detect-daemon/backbone.onnx" \
      --logo-dir "$HOME/.cache/tv-detect-daemon/logos" \
      --output "$D/out/head.bin" \
      --hls-root "$BASIS/snapshot" \
      --surface-uncertain 6 \
      --with-logo --with-audio --with-minute-prior --with-self-training \
      --train-archive "$D/archive" \
      --head-arch mlp32 \
      --prod-seeds 3 \
      --herkunft-belegt \
      --audio-dynamik \
      --cluster-anker aus \
      --archiv-ausschluss "$REPO/docs/archiv-ausschluss-o25.json" \
      --ocr-spalten \
      --stichtag "$STICHTAG" \
      ${ZUSATZ[@]+"${ZUSATZ[@]}"} \
      >"$D/lauf.log" 2>&1 || RC=1
  echo "$(date +%T) Arm $ARM fertig (rc=$RC)"
  grep -E "SigLIP-Spalten:|head-to-head|reason:|PRODUCTION METRIC|GOLDEN-EVAL|Golden-Boden|DEPLOYED|NOT DEPLOYED|Gipfel" "$D/lauf.log" | tail -12
done
exit $RC
