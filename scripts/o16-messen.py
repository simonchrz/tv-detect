#!/usr/bin/env python3
"""O16-Messung: Kantenfehler des Modells gegen den KORRIGIERTEN Golden-Satz.

Registrierte Bedingungen (docs/o16-voller-kopf-preregistration.md):
  1. Anteil Kanten ueber 10 s sinkt um >= 5 Punkte  (Basis 23 %  -> Ziel <= 18 %)
  2. Median verschlechtert sich um <= 0.5 s          (Basis 2.0 s -> Ziel <= 2.5 s)
Waechter: Golden-IoU faellt nicht weiter als die Seed-Streuung.

⚠️ Labels kommen vom GATEWAY, nicht aus dem Snapshot — der ist ein Abzug und
war nach der Korrektur vom 17.08. veraltet (siehe golden-audit.py).
"""
import importlib.util, json, pathlib, ssl, urllib.request, statistics as st, sys
spec = importlib.util.spec_from_file_location("ar", "/Users/simon/src/tv-detect/scripts/agent-review.py")
ar = importlib.util.module_from_spec(spec); spec.loader.exec_module(ar)
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
S = pathlib.Path("/tmp/tv-train-snapshot")

def bl(x):
    a = x.get("ads") if isinstance(x, dict) else x
    return [(float(p[0]), float(p[1])) for p in a] if a else []

def label(u):
    try:
        r = urllib.request.Request(f"{ar.GATEWAY}/recording/{u}/ads")
        with urllib.request.urlopen(r, context=ctx, timeout=20) as x:
            return bl(json.loads(x.read()))
    except Exception:
        return []

def modell(u):
    p = S/f"_rec_{u}"/"ads.json"
    return bl(json.loads(p.read_text())) if p.is_file() else []

fehler, je_kanal = [], {}
n_rec = 0
for u in sorted(ar.golden_uuids()):
    if not (ar.QUELLE/f"{u}.ts").is_file():
        continue
    wahr, mod = label(u), modell(u)
    if not wahr or not mod:
        continue
    n_rec += 1
    k = u[4:u.rfind("-")]
    for l in wahr:
        m = min(mod, key=lambda y: abs(y[0]-l[0]))
        if abs(m[0]-l[0]) > 180:
            continue
        for j in (0, 1):
            e = abs(m[j]-l[j]); fehler.append(e); je_kanal.setdefault(k, []).append(e)

if not fehler:
    print("keine Kanten messbar"); sys.exit(1)
q = st.quantiles(fehler, n=100)
ueber10 = 100*sum(1 for x in fehler if x > 10)/len(fehler)
print(f"{n_rec} Aufnahmen, {len(fehler)} Kanten")
print(f"  Median {st.median(fehler):5.1f} s   p75 {q[74]:5.1f} s   p90 {q[89]:5.1f} s   max {max(fehler):5.1f} s")
print(f"  <=2 s {100*sum(1 for x in fehler if x<=2)/len(fehler):4.0f} %   "
      f"<=5 s {100*sum(1 for x in fehler if x<=5)/len(fehler):4.0f} %   "
      f">10 s {ueber10:4.1f} %")
print()
print("=== O16-Bedingungen (Basis: Median 2.0 s, >10 s 23 %) ===")
b1 = ueber10 <= 18.0
b2 = st.median(fehler) <= 2.5
print(f"  [1] >10 s {ueber10:.1f} %  (Ziel <= 18 %)      {'ERFUELLT' if b1 else 'VERFEHLT'}")
print(f"  [2] Median {st.median(fehler):.1f} s (Ziel <= 2.5 s)   {'ERFUELLT' if b2 else 'VERFEHLT'}")
print(f"\n  O16: {'ERFUELLT' if (b1 and b2) else 'NICHT ERFUELLT'}")
