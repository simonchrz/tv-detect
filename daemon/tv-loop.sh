#!/bin/bash
# Der Fahrer der Verbesserungs-Schleife. Laeuft taeglich um 08:07, also nach
# dem Nightly (03:30, endet gegen 04:10) und nach dem Label-Backup (04:30).
#
# Die Schleife selbst ist NICHT dieses Skript. Hier steht nur, was jeden Tag
# passieren muss, damit ueberhaupt jemand hinsieht: Sensor lesen, Registrierung
# pruefen, Ergebnismass messen — und dann Claude auf den Zustand ansetzen.
#
# Warum lokal und nicht als Cloud-Routine: Nightly-Log, Feature-Cache und
# Trainings-Archiv liegen auf diesem Mac. Ein Agent ohne Zugriff darauf kann
# den Zustand nicht lesen.
#
# Warum ueberhaupt ein Fahrer: die Entscheidungs-Ebene lief bis 2026-08-09 nur,
# wenn jemand danach gefragt hat. Eine Serie, die niemand ansieht, ist keine
# Serie — sie ist eine Datei, die waechst.
LOG="$HOME/Library/Logs/tv-loop.log"
exec >>"$LOG" 2>&1
echo ""
echo "=== $(date '+%F %T') ==="

# launchd startet Agenten mit leerem PATH; claude, python3 und git haengen
# daran (dieselbe Falle wie in tv-train-head.sh).
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin"

REPO="$HOME/src/tv-detect"
PY="$HOME/ml/tv-classifier/.venv/bin/python"
cd "$REPO" || { echo "kein $REPO — Abbruch"; exit 1; }

# ── 1. Zustand einsammeln ───────────────────────────────────────────────
# Getrennt eingesammelt und in eine Datei gelegt, statt Claude die Werkzeuge
# selbst aufrufen zu lassen: so steht im Log, WAS er gesehen hat, auch wenn
# die Bewertung danebengeht.
BERICHT=$(mktemp -t tv-loop)
trap 'rm -f "$BERICHT"' EXIT

{
  "$PY" scripts/loop-status.py 2>&1
  echo ""
  "$PY" scripts/audit-preregistration.py 2>&1
  AUDIT_RC=$?
  echo ""
  echo "(audit exit=$AUDIT_RC — 1 bedeutet Regel verletzt oder Integritaetsproblem)"
  echo ""
  # Das Ergebnismass braucht den Pi. Faellt er aus, ist das kein Grund, den
  # Rest nicht anzusehen.
  "$PY" scripts/review-effort.py 2>&1 || echo "(Korrekturaufwand nicht messbar)"
} >"$BERICHT"

cat "$BERICHT"

# --nur-bericht: Zustand einsammeln und aufhoeren. Zum Nachsehen von Hand und
# um den Agenten zu pruefen, ohne einen Claude-Durchgang auszuloesen.
if [ "$1" = "--nur-bericht" ]; then
  echo "(--nur-bericht: keine Bewertung)"
  exit 0
fi

# ── 2. Bewerten lassen ──────────────────────────────────────────────────
# Der Prompt sagt bewusst NICHT "verbessere das Modell". Er sagt: lies den
# Zustand, halte dich an die Regeln, und tu meistens nichts. Eine Schleife,
# die jeden Tag etwas aendern will, produziert Bewegung statt Fortschritt.
PROMPT=$(cat <<'ENDE'
Du bist der taegliche Durchgang der Verbesserungs-Schleife von tv-detect.

Lies ZUERST docs/experiment-ledger.md — besonders die Entscheidungsregeln
(§1), die offenen Fragen (§3), den Friedhof (§4) und die Leitplanken (§5).
Der Friedhof ist bindend: was dort steht, wird nicht neu vorgeschlagen.

Der Zustandsbericht von heute liegt unter dem Pfad, der dir als erstes
Argument genannt wird. Er enthaelt Sensor, Registrierungs-Audit und das
Ergebnismass.

Deine Aufgabe, in dieser Reihenfolge:

1. Stimmt etwas nicht? Audit-Exit 1, verworfene Naechte, "ALLE Naechte
   verworfen", ein Lauf der gar nicht stattfand, Warnungen zu set_hash oder
   Decoder. Defekte haben Vorrang vor Erkenntnissen.
2. Ist eine laufende Serie reif? Nur dann entscheiden, und nur nach der
   vorab registrierten Regel — nicht nach dem, was die Zahlen nahelegen.
3. Sonst: nichts tun und das auch so sagen. Ein Zwischenstand ist kein
   Ergebnis. Die meisten Tage enden hier, und das ist richtig.

Was du NIE tust, auch nicht mit guter Begruendung: den Golden-Boden senken,
Labels anfassen, am Deploy-Gate vorbei deployen, aus einem Handlauf ins echte
Trainings-Archiv schreiben, den Header-Vertrag oder den Go-Inferenzpfad ohne
Paritaets-Fixture aendern, Aufnahmen loeschen.

Schreib hoechstens 15 Zeilen. Wenn nichts zu tun ist, schreib zwei.
ENDE
)

claude -p "$PROMPT

Zustandsbericht: $BERICHT" 2>&1
echo "loop exit=$?"

# ── 3. Tagesbericht zustellen ───────────────────────────────────────────
# Bis 2026-08-12 landete alles nur in diesem Protokoll — also musste jemand
# danach FRAGEN, wie das Training lief. Genau das soll wegfallen.
#
# Zwei Wege, mit Absicht: die Langfassung an einem festen Ort zum Nachlesen,
# und eine kurze Push-Meldung, die mit dem fuehrt, was zu TUN ist, nicht mit
# Zahlen. Ein taeglicher Bericht, der eine Zahl in den Vordergrund stellt,
# erzieht zum Deuten von Rauschen.
BERICHT_DATEI="$HOME/Library/Logs/tv-loop-tagesbericht.txt"
"$PY" scripts/tagesbericht.py >"$BERICHT_DATEI" 2>&1 || true
cat "$BERICHT_DATEI"

# ⚠️ Der Push geht ueber die PI, nicht von hier. Home Assistant laeuft im
# Host-Netz der Pi und ist vom Mac aus nicht erreichbar (Connection refused) —
# das steht so im Kopf von secrets.env und hat mich beim Bauen prompt einen
# Fehlversuch gekostet.
#
# ⚠️ Text ueber stdin, nicht als Argument: der Bericht enthaelt Klammern und
# Sonderzeichen, und ueber ssh zerbricht das an den Zitierregeln der
# entfernten Shell.
#
# ⚠️ HA_NOTIFY_IPHONE traegt das Praefix "notify." bereits — bleibt es in der
# URL stehen, antwortet HA mit 400.
if "$PY" scripts/tagesbericht.py --kurz 2>/dev/null | head -6 \
   | ssh -o ConnectTimeout=10 raspberrypi5lan 'set -a; . /home/simon/.config/tv-stack/secrets.env; set +a; python3 -c "
import json, os, sys, urllib.request
text = sys.stdin.read().strip()
dienst = os.environ.get(\"HA_NOTIFY_IPHONE\", \"\")
if dienst.startswith(\"notify.\"): dienst = dienst[len(\"notify.\"):]
if not dienst or not os.environ.get(\"HA_TOKEN\"): sys.exit(2)
daten = json.dumps({\"title\": \"tv-detect Schleife\", \"message\": text}).encode()
req = urllib.request.Request(\"http://localhost:8123/api/services/notify/\" + dienst,
                             data=daten,
                             headers={\"Authorization\": \"Bearer \" + os.environ[\"HA_TOKEN\"],
                                      \"Content-Type\": \"application/json\"})
with urllib.request.urlopen(req, timeout=10) as r:
    print(\"Push zugestellt, HTTP\", r.status)
"'; then
  :
else
  # Zustellung darf den Lauf nie zum Scheitern bringen: der Bericht ist das
  # Ergebnis, die Meldung nur der Bote.
  echo "(Push nicht zugestellt — Bericht liegt trotzdem unter $BERICHT_DATEI)"
fi
