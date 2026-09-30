#!/usr/bin/env python3
"""Review-Vorlage — NUR BERICHT, schreibt keine Labels (L1).

Zwei Listen fuer den Menschen:
  1. "Anker sagt Werbung, Label sagt Sendung": Strecken (>= --min-s), an denen
     ein Wiederholungs-Anker (tvd-wiederholung/korpus, Spots, die in vielen
     Aufnahmen wiederkehren) in einer MENSCHLICH gelabelten Aufnahme als
     Sendung steht. Das Fehlerbudget zaehlt diese Sekunden als Label-Seite
     (heute ~1500 s); Memory wiederholte_folgen_sind_werbung: meist echte
     Labelluecken. Je Strecke ein Bildblatt (6 Bilder), Familiengroesse,
     ob die Aufnahme in der App noch reviewbar ist (VOD auf dem Pi).
  2. test-Aufnahmen ohne menschliches Label, die in der App noch reviewbar
     sind: jede davon hebt den belegten Golden-/Test-Kern.

Bilder aus der lokalen Quelle, sonst aus dem HLS-VOD auf dem Pi (Segment
per curl mit --ca, Bild lokal geschnitten). Aufruf:
  review-vorlage.py --ca <caddy-root.pem> --aus review-vorlage.html
"""
import argparse
import base64
import html
import io
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

HIER = Path(__file__).resolve().parent
C = Path.home() / ".cache"
SNAP = Path("/tmp/tv-train-snapshot")
SRC = C / "tv-detect-daemon/source"
ANKER = C / "tvd-wiederholung/korpus"
ARCH = C / "tvd-train-archive"
PI_REC = "http://raspberrypi5lan:9984/recording/{u}/index.m3u8"
PI_SEG = "https://raspberrypi5lan:8443/hls/_rec_{u}/{s}"
TMP = Path("/tmp/review-vorlage")
VF = "scale=iw*sar:ih,scale=640:-2"
sys.path.insert(0, str(HIER))
import label_herkunft as lh  # noqa: E402


def maske(bl, n):
    m = np.zeros(n, bool)
    for s, e in bl:
        m[max(0, int(s)):min(n, int(np.ceil(e)))] = True
    return m


def laeufe(m, min_len):
    out, s = [], None
    for t, v in enumerate(m):
        if v and s is None:
            s = t
        if not v and s is not None:
            if t - s >= min_len:
                out.append((s, t))
            s = None
    if s is not None and len(m) - s >= min_len:
        out.append((s, len(m)))
    return out


_pl = {}


def vod_playlist(u):
    if u not in _pl:
        try:
            with urllib.request.urlopen(PI_REC.format(u=u), timeout=10) as r:
                txt = r.read().decode()
        except Exception:
            txt = ""
        segs, start, dauer = [], 0.0, None
        for l in txt.splitlines():
            if l.startswith("#EXTINF:"):
                dauer = float(l[8:].split(",")[0])
            elif l and not l.startswith("#") and dauer is not None:
                segs.append((start, start + dauer, l.strip()))
                start += dauer
                dauer = None
        _pl[u] = segs
    return _pl[u]


def bild(u, t, ca):
    p = TMP / f"{u}_{int(t)}.jpg"
    if p.is_file():
        return p
    q = SRC / f"{u}.ts"
    if q.is_file():
        subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-ss", f"{t:.1f}", "-i", str(q),
                        "-frames:v", "1", "-vf", VF, str(p)])
    else:
        for a, b, s in vod_playlist(u):
            if a <= t < b:
                seg = TMP / f"{u}_{s}"
                if not seg.is_file():
                    cmd = ["curl", "-s", "-f", "-m", "30", "-o", str(seg), PI_SEG.format(u=u, s=s)]
                    if ca:
                        cmd[1:1] = ["--cacert", str(ca)]
                    subprocess.run(cmd)
                if seg.is_file():
                    subprocess.run(["ffmpeg", "-v", "quiet", "-y", "-ss", f"{t - a:.2f}", "-i", str(seg),
                                    "-frames:v", "1", "-vf", VF, str(p)])
                break
    return p if p.is_file() else None


def blatt(pfade):
    ims = [Image.open(p).convert("RGB").resize((256, 144)) for p in pfade]
    sh = Image.new("RGB", (256 * 3, 144 * 2))
    for j, im in enumerate(ims):
        sh.paste(im, ((j % 3) * 256, (j // 3) * 144))
    b = io.BytesIO()
    sh.save(b, "JPEG", quality=70)
    return base64.b64encode(b.getvalue()).decode()


def mmss(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def meta_von(u):
    try:
        return json.loads(str(np.load(ARCH / f"{u}.npz", allow_pickle=True)["meta"]))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ca", help="Caddy-Root-Zertifikat fuer die Segmente vom Pi")
    ap.add_argument("--aus", default="review-vorlage.html")
    ap.add_argument("--min-s", type=int, default=10)
    ap.add_argument("--max-strecken", type=int, default=80)
    a = ap.parse_args()
    TMP.mkdir(exist_ok=True)
    led = json.loads((ARCH / "split-ledger.json").read_text())
    led = led.get("eimer", led)

    # ── Liste 1 ────────────────────────────────────────────────────────
    strecken = []
    for d in sorted(SNAP.glob("_rec_*")):
        u = d.name[5:]
        f = d / "ads_user.json"
        if not f.is_file():
            continue
        raw = json.loads(f.read_text())
        if lh.mensch_aus_markern(raw) is not True:
            continue
        ak = ANKER / f"{u}.json"
        if not ak.is_file():
            continue
        anker = json.loads(ak.read_text()).get("anchored", [])
        if not anker:
            continue
        mensch = [(float(s), float(e)) for s, e in (raw.get("ads") or [])]
        n = int(max([x["end_s"] for x in anker] + [e for _, e in mensch] + [0])) + 1
        A = maske([(x["window_start_s"], x["end_s"]) for x in anker], n)
        T = maske(mensch, n)
        for s, e in laeufe(A & ~T, a.min_s):
            fam = max((x["family_size"] for x in anker
                       if x["window_start_s"] < e and x["end_s"] > s), default=0)
            strecken.append(dict(uuid=u, a=s, b=e, fam=fam, eimer=led.get(u, "?"),
                                 titel=meta_von(u).get("title", ""), mensch=mensch))
    print(f"{len(strecken)} Strecken (>= {a.min_s} s) in "
          f"{len({s['uuid'] for s in strecken})} menschlich gelabelten Aufnahmen; "
          f"{sum(s['b'] - s['a'] for s in strecken)} s gesamt", flush=True)
    strecken.sort(key=lambda s: (-(s["b"] - s["a"]) * min(s["fam"], 50), s["uuid"]))
    reviewbar = {}
    for s in strecken:
        u = s["uuid"]
        if u not in reviewbar:
            reviewbar[u] = bool(vod_playlist(u))
    strecken.sort(key=lambda s: (not reviewbar[s["uuid"]], -(s["b"] - s["a"]) * min(s["fam"], 50)))
    zeilen = []
    for s in strecken[:a.max_strecken]:
        rand = 0.05 * (s["b"] - s["a"])
        pfade = [bild(s["uuid"], t, a.ca) for t in np.linspace(s["a"] + rand, s["b"] - rand, 6)]
        img = (f'<img alt="6 Bilder aus der Strecke" src="data:image/jpeg;base64,{blatt(pfade)}">'
               if all(pfade) else "<p><em>keine Bilder (Quelle und VOD fehlen)</em></p>")
        lab = ", ".join(f"{mmss(x)}–{mmss(y)}" for x, y in s["mensch"]) or "keine"
        zeilen.append(f"""<article class="k {'ja' if reviewbar[s['uuid']] else 'nein'}">
<header><h2>{html.escape(s['titel'])}</h2><span class="u">{s['uuid']} · {s['eimer']}</span></header>
<p><b>{mmss(s['a'])}–{mmss(s['b'])}</b> ({s['b'] - s['a']} s) · Anker-Familie bis {s['fam']} Aufnahmen ·
{'in der App reviewbar' if reviewbar[s['uuid']] else 'nicht mehr auf dem Pi (nur Archiv)'}</p>
{img}<p class="lab">Label-Werbung bisher: {lab}</p></article>""")

    # ── Liste 2 ────────────────────────────────────────────────────────
    offen = []
    for u, e in sorted(led.items()):
        if e != "test":
            continue
        f = SNAP / f"_rec_{u}" / "ads_user.json"
        try:
            if lh.mensch_aus_markern(json.loads(f.read_text())) is True:
                continue
        except Exception:
            pass
        if vod_playlist(u):
            m = meta_von(u)
            offen.append((u, m.get("title", ""), m.get("slug", u.split("-")[1] if u.startswith("dvr-") else "")))
    print(f"{len(offen)} reviewbare test-Aufnahmen ohne menschliches Label", flush=True)
    liste2 = "".join(f"<li><b>{html.escape(t)}</b> <span class='u'>{u} · {sl}</span></li>" for u, t, sl in offen)

    seite = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Review-Vorlage</title><style>
:root{{--bg:#f6f5f2;--fg:#1d1d1b;--mut:#6b6a66;--karte:#fff;--rand:#e2e0da;--ja:#1f7a3a;--nein:#9d9b95}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#161615;--fg:#ecebe7;--mut:#9d9b95;--karte:#1f1f1d;--rand:#33322f;--ja:#7fd39a}}}}
:root[data-theme="dark"]{{--bg:#161615;--fg:#ecebe7;--mut:#9d9b95;--karte:#1f1f1d;--rand:#33322f;--ja:#7fd39a}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0;padding:24px 16px;max-width:820px;margin-inline:auto}}
h1{{font-size:22px;margin:0 0 6px}}h2{{font-size:17px;margin:0}}.intro{{color:var(--mut)}}
.k{{background:var(--karte);border:1px solid var(--rand);border-radius:10px;padding:14px;margin:0 0 14px}}
.k header{{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}}.u{{color:var(--mut);font:12px ui-monospace,monospace}}
.k p{{margin:6px 0}}.ja h2{{color:var(--ja)}}.k img{{width:100%;height:auto;border-radius:6px;display:block;margin:8px 0}}
.lab{{color:var(--mut);font-size:13px}}ol li{{margin:4px 0}}
</style></head><body>
<h1>Review-Vorlage</h1>
<p class="intro">Nur eine Vorlage: nichts wurde geändert. Stand des Label-Schnappschusses:
{time.strftime('%d.%m.%Y %H:%M', time.localtime(SNAP.stat().st_mtime))}.</p>
<h2>1. Wiederholungs-Anker im Label als Sendung ({len(strecken)} Strecken, die {min(len(strecken), a.max_strecken)} gewichtigsten)</h2>
<p class="intro">Spots, die in vielen Aufnahmen wiederkehren, stehen hier als Sendung. Grün = in der App reviewbar.</p>
{''.join(zeilen)}
<h2>2. test-Aufnahmen ohne menschliches Label, in der App reviewbar ({len(offen)})</h2>
<p class="intro">Jede reviewte hebt den belegten Kern des Maßstabs.</p><ol>{liste2}</ol>
</body></html>"""
    Path(a.aus).write_text(seite)
    print(f"Bericht: {a.aus}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
