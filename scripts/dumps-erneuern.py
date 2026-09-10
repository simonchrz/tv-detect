#!/usr/bin/env python3
"""Signal-Dumps des Messsatzes erneuern, ohne ein einziges Label anzufassen.

WARUM
-----
Das Fehlerbudget (`fehlerbudget.py`) misst ueber `--replay-signals`, und
ein Signal-Dump enthaelt die NN-Ausgaben EINGEFROREN. Am 2026-09-08 waren
58 der 98 Dumps vom 15.08., der Kopf hatte seitdem mehrfach gewechselt.
Die Aufteilung der Verlustursachen beschrieb also einen Kopf, der nicht
mehr lief -- und ausgerechnet der groesste Posten ("NN erfindet", 39 %
von 5882 Verlustsekunden) ist der, an dem noch Spielraum ist.

WARUM NICHT EINFACH NEU DETECTEN
--------------------------------
Ein Dump entsteht nur beim Detect, und der Detect schreibt die Bloecke
neu. Die 98 Aufnahmen des Messsatzes sind genau die MIT menschlichem
Label. Ein Redetect wuerde unter dem Menschen die Bloecke austauschen,
gegen die er geurteilt hat. Der Daemon warnt an seiner Dump-Stelle
woertlich davor. Leitplanke: Labels sind Eingabe, keine Stellschraube.

Deshalb laeuft das hier ueber `process_detect(..., nur_dump=True)` --
DERSELBE Code, DIESELBE Kommandozeile, aber der Lauf endet vor jeder
Schreiboperation. Ein nachgebautes Kommando waere der andere Ausweg und
ist der schlechtere: die nackten Vorgaben des Binaries weichen von der
Produktion ab, und genau so las die Kanten-Messung vom 2026-07-24 +0.016,
wo die treue -0.015 las.

WAS DANACH MOEGLICH IST
-----------------------
Alt gegen neu auf identischen Labels und identischem Dekoder ist eine
GEPAARTE Kopf-Messung ueber 98 Aufnahmen. Unverzerrt sind davon nur die
Testaufnahmen -- der Rest lag im Training des Kopfes. Beides getrennt
ausweisen, nie zusammen.

⚠️ DER KOPF MUSS UEBER DIE GANZE KAMPAGNE DERSELBE SEIN. Sechs Stunden
Laufzeit ueberspannen die naechtliche Ausbildung um 03:30. Wechselt der
Kopf mittendrin, ist der Messsatz eine Mischung aus zwei Modellen und
sieht trotzdem sauber aus. Deshalb wird der Fingerabdruck von head.bin
zu Beginn genommen und vor JEDER Aufnahme geprueft.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DAEMON = Path(__file__).resolve().parent.parent / "daemon" / "tv-thumbs-daemon.py"
MESSSATZ = Path.home() / ".cache/tvd-train-archive/messsatz-2026-09-07.json"
MODELL = Path.home() / ".cache/tv-detect-daemon"


PLIST = (Path.home() / "Library/LaunchAgents"
         / "com.user.tv-thumbs-daemon.plist")


def umgebung_vom_daemon():
    """Die Umgebung des laufenden Daemons uebernehmen, VOR dem Import.

    ⚠️ SELBST HINEINGELAUFEN, 2026-09-08. Der erste Probelauf las
    "speaker aus (SPEAKER_ENABLE=0)" und teilte in 12 Stuecke; der Daemon
    laeuft mit SPEAKER_ENABLE=1 und DETECT_PARALLEL=3, also 4 Stuecke.
    Derselbe Code, dieselbe Kommandozeile -- und trotzdem ein anderer
    Lauf, weil die Konstanten auf Modulebene beim IMPORT aus der Umgebung
    gelesen werden. Genau die Sorte Abweichung, gegen die dieses Skript
    gebaut ist, nur eine Ebene tiefer als erwartet.

    Quelle ist der launchd-Eintrag, nicht eine abgeschriebene Liste:
    was dort steht, ist per Definition das, womit der Daemon laeuft.
    """
    try:
        aus = subprocess.run(
            ["/usr/libexec/PlistBuddy", "-c", "Print :EnvironmentVariables",
             str(PLIST)], capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        print(f"⚠️ launchd-Umgebung nicht lesbar ({e}) — ABBRUCH, ein Lauf "
              f"mit anderer Umgebung misst etwas anderes als die Produktion.")
        raise SystemExit(3)
    gesetzt = {}
    for ln in aus.splitlines():
        if "=" not in ln:
            continue
        k, _, v = ln.partition("=")
        k, v = k.strip(), v.strip()
        if k and not k.startswith("{") and not k.startswith("}"):
            os.environ[k] = v
            gesetzt[k] = v
    if not gesetzt:
        print("⚠️ launchd-Eintrag nennt keine Umgebung — ABBRUCH.")
        raise SystemExit(3)
    return gesetzt


def einzelstueck(ziel):
    """Nur EINE Kampagne je Zielverzeichnis. Gibt die Sperrdatei zurueck.

    ⚠️ SELBST HINEINGELAUFEN, 2026-09-10. Ich hatte um 06:34 von Hand
    eine Kampagne gestartet; um 08:2x startete der naechtliche Lauf eine
    zweite auf demselben Verzeichnis. Die zweite raeumte die Dumps der
    ersten weg (aus ihrer Sicht hatte der Kopf gewechselt), beide
    schrieben in dieselbe Protokolldatei und ueberschrieben sich, und am
    Ende stand ein Dump von 98 da, ohne dass etwas abgestuerzt waere.

    Der Lock ist bewusst eine Datei mit PID und nicht flock: er soll
    auch nach einem harten Abbruch lesbar sein und sagen, WER hier
    arbeitet. Eine Sperre mit totem Prozess wird uebernommen.
    """
    sperre = ziel / ".laeuft"
    if sperre.is_file():
        try:
            pid = int(sperre.read_text().split()[0])
        except Exception:
            pid = None
        lebt = False
        if pid:
            try:
                os.kill(pid, 0)
                lebt = True
            except OSError:
                lebt = False
        if lebt:
            print(f"Es laeuft bereits eine Kampagne auf diesem Verzeichnis "
                  f"(PID {pid}). Nichts zu tun.")
            return None
        print(f"Verwaiste Sperre von PID {pid} uebernommen.")
    sperre.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    return sperre


def modelle_frisch(mod):
    """Kopf und Beilagen vom Gateway holen, BEVOR der Abdruck faellt.

    ⚠️ SELBST HINEINGELAUFEN, 2026-09-10, erste Nacht mit dieser
    Kampagne. Sie las den Fingerabdruck aus dem MODELL-CACHE des Mac,
    und der erneuert sich nur beim naechsten Detect. Der Pi trug um
    04:16 den frisch deployten Kopf, der Cache um 06:17 noch den von
    gestern — die Kampagne meldete folgerichtig "nichts zu tun" und tat
    damit genau das Falsche. Ein Nachfuehren, das seine Referenz aus
    einem Zwischenspeicher zieht, fuehrt nichts nach.

    Der Abruf ist ausserdem nuetzlich: er waermt denselben Cache, aus
    dem die Detects gleich lesen.
    """
    geholt = []
    for name in ("head.bin", "head.audio.json", "head.calibration.json",
                 "head.channel-map.json", "head.minute-prior.json"):
        ziel = mod.MODEL_CACHE / name
        vorher = ziel.stat().st_mtime if ziel.is_file() else 0
        try:
            mod.http_download(
                f"{mod.GATEWAY}/api/internal/detect-models/{name}", ziel)
        except Exception as e:
            if "404" not in str(e):
                print(f"  {name} nicht geholt ({e})")
            continue
        if ziel.is_file() and ziel.stat().st_mtime != vorher:
            geholt.append(name)
    if geholt:
        print("frisch vom Gateway geholt:", ", ".join(geholt))
    return geholt


def daemon_laden():
    """Den Daemon als Modul laden, ohne seine Schleifen zu starten.

    Sicher, weil alle Threads in main() unter `if __name__ == "__main__"`
    haengen; auf Modulebene stehen nur Konstanten und zwei mkdir.
    """
    spec = importlib.util.spec_from_file_location("tvthumbs", DAEMON)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kopf_abdruck():
    p = MODELL / "head.bin"
    if not p.is_file():
        return None
    h = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
    beilage = MODELL / "head.audio.json"
    ad = " ".join(beilage.read_text().split()) if beilage.is_file() else "(keine)"
    return h, ad


def andere_arbeit_laeuft():
    """Laeuft gerade ein Detect oder eine Ausbildung? Dann warten.

    Die Kampagne ist Forschung und hat Vorrang vor NICHTS. Sie darf weder
    die naechtliche Ausbildung noch einen echten Detect verlangsamen.
    """
    try:
        out = subprocess.run(["ps", "-eo", "command"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return False        # im Zweifel weiterlaufen, nicht haengenbleiben
    eigen = f"dumps-erneuern"
    for ln in out.splitlines():
        if eigen in ln:
            continue
        if "/tv-detect " in ln or ln.rstrip().endswith("/tv-detect"):
            return "ein Detect"
        if "train-head.py" in ln:
            return "die Ausbildung"
    return False


def gateway_wartet(mod, hoechstens_s=1800):
    """Warten, bis das Gateway wieder antwortet. False = aufgegeben."""
    import urllib.request
    gewartet = 0
    while gewartet < hoechstens_s:
        try:
            with urllib.request.urlopen(f"{mod.GATEWAY}/healthz", timeout=10,
                                        context=mod.CTX) as r:
                if r.status == 200:
                    if gewartet:
                        print(f"  Gateway wieder da nach {gewartet}s",
                              flush=True)
                    return True
        except Exception:
            pass
        if gewartet == 0:
            print("  Gateway stumm — warte, statt Aufnahmen zu verbrennen",
                  flush=True)
        time.sleep(20)
        gewartet += 20
    print(f"  Gateway seit {hoechstens_s}s stumm — Abbruch dieser Aufnahme",
          flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ziel", default=str(MODELL / "emit-signals-neu"),
                    help="Verzeichnis fuer die neuen Dumps")
    ap.add_argument("--messsatz", default=str(MESSSATZ))
    ap.add_argument("--limit", type=int, default=0,
                    help="nur die ersten N Aufnahmen (Probelauf)")
    ap.add_argument("--ruecksichtslos", action="store_true",
                    help="nicht auf Detects/Ausbildung warten")
    ap.add_argument("--trocken", action="store_true",
                    help="nur zeigen, was liefe")
    a = ap.parse_args()

    umg = umgebung_vom_daemon()
    print("Umgebung vom Daemon:",
          " ".join(f"{k}={v}" for k, v in sorted(umg.items())))

    satz = json.loads(Path(a.messsatz).read_text())
    uuids = satz["uuids"]
    ziel = Path(a.ziel)
    ziel.mkdir(parents=True, exist_ok=True)
    sperre = None
    if not a.trocken:
        sperre = einzelstueck(ziel)
        if sperre is None:
            return 0

    # ⚠️ REIHENFOLGE. Modul laden, Modelle holen, DANN den Abdruck
    # nehmen. Andersherum beschreibt der Abdruck den Cache-Stand von
    # gestern (siehe modelle_frisch).
    mod = daemon_laden()
    if a.trocken:
        # Ein Trockenlauf fasst den Modell-Cache nicht an. Dann kann der
        # Abdruck unten aber der von gestern sein — sagen statt schweigen.
        print("⚠️ Trockenlauf: Modelle werden NICHT geholt, der "
              "Kopf-Abdruck unten kann veraltet sein.")
    else:
        modelle_frisch(mod)
    abdruck = kopf_abdruck()
    if not abdruck:
        print("head.bin fehlt im Modell-Cache — nichts zu messen.")
        return 1
    h0, ad0 = abdruck
    print(f"Kopf {h0}, Audio-Beilage {ad0}")
    print(f"Messsatz {satz.get('name')} ({satz.get('hash')}), {len(uuids)} Aufnahmen")
    print(f"Ziel {ziel}")

    # ⚠️ WECHSELT DER KOPF, IST DER GANZE SATZ WERTLOS. Ein Dump friert
    # die NN-Ausgaben ein; ein Verzeichnis mit Dumps aus zwei Modellen
    # sieht vollstaendig aus und ist eine Mischung. Deshalb steht der
    # Fingerabdruck NEBEN den Dumps, und bei Abweichung wird geraeumt
    # statt ergaenzt. Das macht den Aufruf idempotent: das Skript kann
    # jede Nacht blind starten, tut bei unveraendertem Kopf nichts und
    # baut nach einem Deploy neu auf.
    kopf_datei = ziel / ".kopf"
    frueher = kopf_datei.read_text().strip() if kopf_datei.is_file() else None
    if frueher and frueher != h0:
        weg = sorted(ziel.glob("*.json"))
        if a.trocken:
            # ⚠️ Ein Trockenlauf fasst NICHTS an. Der erste Entwurf
            # raeumte auch hier, weil der Ausstieg fuer --trocken weiter
            # unten steht — ein Probelauf haette den Satz geloescht.
            print(f"Kopf hat gewechselt ({frueher} -> {h0}) — "
                  f"WUERDE {len(weg)} Dumps raeumen (Trockenlauf)")
        else:
            for f in weg:
                try: f.unlink()
                except Exception: pass
            print(f"Kopf hat gewechselt ({frueher} -> {h0}) — "
                  f"{len(weg)} Dumps des alten Kopfes geraeumt")
    if not a.trocken:
        kopf_datei.write_text(h0)

    # Fortsetzbar: was schon da ist, wird nicht neu gerechnet.
    offen = [u for u in uuids if not (ziel / f"{u}.json").is_file()]
    print(f"offen: {len(offen)}, fertig: {len(uuids)-len(offen)}")
    if a.limit:
        offen = offen[:a.limit]
    if a.trocken:
        for u in offen[:10]:
            print("  wuerde:", u)
        print(f"  ... insgesamt {len(offen)}")
        return 0
    if not offen:
        print("nichts zu tun.")
        return 0

    ok = fehl = 0
    t_start = time.time()
    for i, u in enumerate(offen, 1):
        jetzt = kopf_abdruck()
        if jetzt != abdruck:
            # NICHT weitermachen. Ein gemischter Messsatz sieht sauber aus
            # und ist es nicht.
            if sperre:
                try: sperre.unlink()
                except Exception: pass
            # Kein Aufraeum-Befehl mehr fuer den Menschen: der naechste
            # Start raeumt selbst, weil .kopf nicht mehr passt.
            print(f"\n⚠️ ABBRUCH: der Kopf hat sich geaendert "
                  f"({h0} -> {jetzt[0] if jetzt else 'weg'}). Die {ok} "
                  f"Dumps sind vom alten Kopf. Einfach neu starten — der "
                  f"naechste Lauf raeumt sie selbst weg, weil die "
                  f"Kopf-Marke nicht mehr passt.", flush=True)
            return 2
        if not a.ruecksichtslos:
            gewartet = 0
            while True:
                was = andere_arbeit_laeuft()
                if not was:
                    break
                if gewartet == 0:
                    print(f"  warte, {was} laeuft…", flush=True)
                time.sleep(30)
                gewartet += 30
                if gewartet > 4 * 3600:
                    print("  wartet seit 4 h — mache trotzdem weiter",
                          flush=True)
                    break
        rest = ""
        if ok:
            je = (time.time() - t_start) / ok
            rest = f", Rest ~{je*(len(offen)-i+1)/3600:.1f} h"
        print(f"[{i}/{len(offen)}] {u}{rest}", flush=True)
        # ⚠️ WARTEN STATT DURCHBRENNEN. Am 2026-09-09 um 08:26 war das
        # Gateway drei Minuten weg. Der erste Entwurf hat in dieser Zeit
        # 54 Aufnahmen im Fuenf-Sekunden-Takt als Fehlschlag abgehakt --
        # aus einem Aussetzer wurde ein halber Messsatz. Ein Lauf ueber
        # Stunden MUSS Aussetzer aushalten, sonst misst am Ende die
        # Netzqualitaet mit.
        gut = False
        for versuch in range(1, 4):
            try:
                gut = mod.process_detect(u, nur_dump=True, dump_ziel=str(ziel))
                break
            except Exception as e:
                print(f"  Ausnahme ({versuch}/3): {e}", flush=True)
                if not gateway_wartet(mod):
                    break
        ok, fehl = (ok + 1, fehl) if gut else (ok, fehl + 1)

    if sperre:
        try: sperre.unlink()
        except Exception: pass
    print(f"\nfertig: {ok} Dumps, {fehl} Fehlschlaege, "
          f"{(time.time()-t_start)/3600:.1f} h")
    print(f"Kopf am Ende: {kopf_abdruck()[0]} (Start {h0})")
    return 0 if fehl == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
