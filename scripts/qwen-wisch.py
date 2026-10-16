#!/usr/bin/env python3
"""Qwen3.8-27B wischt Maßstab-Karten in Simons Namen (Simon 2026-10-16).

Läuft mit der venv ~/ml/vlm-exp/.venv (mlx-vlm). Schreibt NIE Labels; es
beantwortet Wischkarten mit `von: qwen`, der tv-recorder vermerkt das als
`geprueft_per: wisch-qwen`, wenn dadurch eine Aufnahme gehoben wird.

Sicherungen (alle müssen halten, sonst wird nichts oder nichts mehr geschickt):
  1. Eichtor vor jedem Lauf: die Kontrollkarten des Prüfsatzes vom 16.10.
     (bekannte Antwort) — jede Karte, die das Modell sicher beantwortet,
     muss stimmen, und mindestens --eich-min davon müssen sicher sein.
  2. Zwei Durchgänge je Karte: A = 5 Bilder 480 px vom Pi (t-6…+6 s),
     B = der Wischstreifen der App (5 × 320 px). Geschickt wird nur, wenn A
     und B dieselbe Antwort geben UND beide mindestens --schwelle sicher sind.
     Alles andere bleibt für Simon liegen.
  3. Antworten in zufälliger Reihenfolge; die erste FALSCHE Kontrollkarte
     beendet den Lauf sofort.
  4. Nur anlass=massstab (wie der Agentenlauf), höchstens --max Karten.
Bericht am Ende: geschickt / liegen gelassen / Kontrollen richtig / gehoben.

Aufruf:
  ~/ml/vlm-exp/.venv/bin/python scripts/qwen-wisch.py --schwelle 0.95 [--trocken]
"""
import argparse
import io
import json
import random
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.home() / "ml/vlm-exp"))
from vlm_wisch import FRAGE, OFFS  # noqa: E402

MODELL = "mlx-community/Qwen3.8-27B-4bit"
API = "http://raspberrypi5lan:9984/api/review/wisch"
EICH = Path.home() / ".cache/tvd-qwen-wisch/eichsatz"   # Bilder + eich.json
LOG = Path.home() / ".cache/tvd-qwen-wisch/laeufe.jsonl"
# Geholte Karten sind 6 h fuer uns reserviert (wischKartenTTL). Bricht ein Lauf
# ab, holt der naechste sie hier wieder ab, statt sie bis zum Ablauf zu sperren.
OFFEN = Path.home() / ".cache/tvd-qwen-wisch/offene-karten.json"
WOERTER = ("WERBUNG", "TRAILER", "SENDUNG")


def http(url, daten=None):
    req = urllib.request.Request(url, data=json.dumps(daten).encode() if daten else None,
                                 headers={"Content-Type": "application/json"} if daten else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


class Modell:
    def __init__(self):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        self.model, self.proc = load(MODELL)
        self.cfg = load_config(MODELL)
        tok = self.proc.tokenizer if hasattr(self.proc, "tokenizer") else self.proc
        self.ids = [tok.encode(w, add_special_tokens=False)[0] for w in WOERTER]
        assert len(set(self.ids)) == 3
        self.cache = {}

    def frage(self, bilder, sender, titel):
        import mlx.core as mx
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        f = FRAGE.format(sender=sender, titel=titel or "unbekannt")
        if f not in self.cache:
            self.cache[f] = apply_chat_template(self.proc, self.cfg, f, num_images=len(bilder))
        for r in stream_generate(self.model, self.proc, self.cache[f], [str(b) for b in bilder],
                                 max_tokens=1, temperature=0.0):
            lp = np.array(r.logprobs.astype(mx.float32)).reshape(-1)
            z = np.array([lp[i] for i in self.ids])
            p = np.exp(z - z.max())
            p /= p.sum()
            return {w.lower(): float(x) for w, x in zip(WOERTER, p)}


def antwort(p):
    w = max(p, key=p.get)
    return w, p[w]


def eichtor(m, schwelle, mindest):
    eich = json.loads((EICH / "eich.json").read_text())
    sicher, falsch = 0, []
    for k in eich:
        bilder = [EICH / f"{k['uuid']}_{k['t'] + o:05d}.jpg" for o in OFFS]
        if not all(b.is_file() for b in bilder):
            continue
        w, c = antwort(m.frage(bilder, k["uuid"][4:].rsplit("-", 1)[0], k.get("titel")))
        if c < schwelle:
            continue
        sicher += 1
        bi = "sendung" if w == "sendung" else "werbung"
        if bi != k["wahr_binaer"]:
            falsch.append((k["uuid"], k["t"], w))
    return sicher, falsch


def ziehe_480(karten, tmp):
    """Pass A: 5 Bilder je Karte, 480 px, vom Pi (gleiches Werkzeug wie die Messung)."""
    specs = " ".join(f"{k['uuid']}:{int(k['t']) - 6}:{int(k['t']) + 6}:3" for k in karten)
    cmd = ("rm -rf /tmp/qwen-wisch && mkdir -p /tmp/qwen-wisch && cd /home/simon/tv-tools && "
           f"ZIEH_OUT=/tmp/qwen-wisch ZIEH_BREITE=480 PYTHONIOENCODING=utf-8 python3 zieh.py {specs} "
           "&& cd /tmp/qwen-wisch && tar cf - .")
    r = subprocess.run(["ssh", "raspberrypi5lan", cmd], capture_output=True, timeout=3600)
    if r.returncode != 0:
        sys.exit("Bilder vom Pi fehlgeschlagen: " + r.stderr.decode()[-300:])
    tarfile.open(fileobj=io.BytesIO(r.stdout)).extractall(tmp)


def streifen_teile(karte, tmp):
    """Pass B: der Wischstreifen der App, in seine 5 Bilder zerlegt."""
    b = Image.open(io.BytesIO(http(f"http://raspberrypi5lan:9984{karte['streifen_url']}"))).convert("RGB")
    n = len(OFFS)
    w = b.width // n
    aus = []
    for i in range(n):
        p = tmp / f"B_{karte['id']}_{i}.jpg"
        b.crop((i * w, 0, (i + 1) * w, b.height)).save(p, quality=90)
        aus.append(p)
    return aus


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schwelle", type=float, required=True)
    ap.add_argument("--eich-min", type=int, default=25)
    ap.add_argument("--max", type=int, default=150)
    ap.add_argument("--trocken", action="store_true", help="entscheiden und berichten, nichts schicken")
    a = ap.parse_args()
    t0 = time.time()
    m = Modell()

    sicher, falsch = eichtor(m, a.schwelle, a.eich_min)
    print(f"Eichtor: {sicher} sichere Kontrollkarten, {len(falsch)} falsch")
    if falsch or sicher < a.eich_min:
        sys.exit(f"Eichtor nicht bestanden ({falsch[:3]}) — nichts geschickt.")

    karten = []
    if OFFEN.is_file():
        alt = json.loads(OFFEN.read_text())
        if time.time() - alt["ts"] < 5.5 * 3600:
            karten = alt["karten"]
            print(f"{len(karten)} Karten aus dem abgebrochenen Lauf wieder aufgenommen")
    while len(karten) < a.max:
        neu = json.loads(http(f"{API}/naechste?n=50&anlass=massstab")).get("karten") or []
        neu = [k for k in neu if k["id"] not in {x["id"] for x in karten}]
        if not neu:
            break
        karten += neu
    karten = karten[:a.max]
    OFFEN.parent.mkdir(parents=True, exist_ok=True)
    OFFEN.write_text(json.dumps({"ts": time.time(), "karten": karten}))
    print(f"{len(karten)} Karten geholt (reserviert 6 h)")
    if not karten:
        return 0
    tmp = Path(tempfile.mkdtemp(prefix="qwen-wisch-"))
    ziehe_480(karten, tmp)

    entscheid = []
    for k in karten:
        sender = k["uuid"][4:].rsplit("-", 1)[0]
        A = [tmp / f"{k['uuid']}_{int(k['t']) + o:05d}.jpg" for o in OFFS]
        if not all(p.is_file() for p in A):
            entscheid.append((k, None, "Bilder fehlen")); continue
        wa, ca = antwort(m.frage(A, sender, k.get("titel")))
        wb, cb = antwort(m.frage(streifen_teile(k, tmp), sender, k.get("titel")))
        if wa != wb:
            entscheid.append((k, None, f"uneinig {wa}/{wb}")); continue
        if min(ca, cb) < a.schwelle:
            entscheid.append((k, None, f"unsicher {min(ca, cb):.2f}")); continue
        entscheid.append((k, wa, f"{ca:.3f}/{cb:.3f}"))

    schicken = [e for e in entscheid if e[1]]
    random.shuffle(schicken)
    bericht = {"ts": int(time.time()), "schwelle": a.schwelle, "geholt": len(karten),
               "sicher": len(schicken), "geschickt": 0, "kontrollen": 0, "kontrollen_richtig": 0,
               "gehoben": [], "abbruch": None, "trocken": a.trocken,
               "liegen": {}}
    for _, w, g in entscheid:
        if not w:
            grund = g.split()[0]
            bericht["liegen"][grund] = bericht["liegen"].get(grund, 0) + 1
    if not a.trocken:
        for k, w, _ in schicken:
            r = json.loads(http(API, {"id": k["id"], "urteil": w, "von": "qwen"}))
            bericht["geschickt"] += 1
            if r.get("gehoben"):
                bericht["gehoben"].append(k["uuid"])
            if r.get("kontrolle"):
                bericht["kontrollen"] += 1
                if r.get("richtig"):
                    bericht["kontrollen_richtig"] += 1
                else:
                    bericht["abbruch"] = f"Kontrolle falsch: {k['uuid']}@{k['t']} {w}, erwartet {r.get('erwartet')}"
                    break
    if not a.trocken:
        OFFEN.unlink(missing_ok=True)   # beantwortet oder liegen gelassen: nichts mehr offen
    bericht["sekunden"] = round(time.time() - t0)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(bericht, ensure_ascii=False) + "\n")
    print(json.dumps(bericht, ensure_ascii=False, indent=1))
    return 1 if bericht["abbruch"] else 0


if __name__ == "__main__":
    sys.exit(main())
