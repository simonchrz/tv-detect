# O13 — Der Schattenlauf: setzt der Bildschirm-Text bessere Kanten?
(Vorab-Registrierung)

**Geschrieben 2026-08-15, bevor eine einzige Schatten-Zeile existiert.**

## Warum diese Frage nicht wie O9–O12 gestellt werden darf

Vier Registrierungen zur selben Idee, vier Fehlschläge an vier
verschiedenen Bedingungen — und beim letzten war der Golden-Satz die
Instanz. **Er ist damit für diese Regel verbraucht.** Ein fünfter
handverlesener Satz wäre genau das Muster, das §3ad benannt hat: wer oft
genug neu registriert, gewinnt irgendwann durch Zufall.

Es gibt trotzdem einen sauberen Weg, und es ist der einzige: **prospektiv**
messen. Der Schattenlauf sammelt auf Aufnahmen, die es beim Schreiben
dieser Registrierung noch nicht gibt und deren Labels noch niemand gesetzt
hat. Kein Satz kann dabei nachträglich passend gewählt werden, weil er
beim Registrieren gar nicht existiert.

## Was sich seit O12 tatsächlich geändert hat

Nicht die Idee, sondern drei Voraussetzungen:

1. **Das Signal ist echt in der Kette.** `--ocr-marker` erhebt es im
   Detect-Lauf (`internal/signals/ocr.go`), nicht mehr ein Python-Skript
   daneben. Kosten gemessen: +7 % Laufzeit.
2. **Die Kennzahl ist die richtige.** O9–O12 maßen Block-IoU — die misst
   überwiegend Block-MASSE. Gemessen wird ab jetzt der **Kantenfehler in
   Sekunden**, weil das die Zielgröße ist (§3af).
3. **Die Referenz ist sauberer.** 10 belegte Label-Fehler wurden heute
   korrigiert, dazu 24 aus früheren Runden.

⚠️ Keine dieser drei Änderungen ist ein Argument, den verbrauchten
Golden-Satz wieder aufzumachen. Sie sind der Grund, die Frage überhaupt
noch zu stellen — nicht der Grund, sie anders zu prüfen.

## Die Regel — fixiert

An jeder hsmm-Blockgrenze im Fenster ±90 s nach OCR-Treffern suchen
(Abtastung 2 s). Ein Treffer zählt nur, wenn dasselbe Muster im
Referenzfenster +180…+240 s NICHT mehr steht (Dauer-Einblendungen, §3ac).
Die Kante wird auf den äußersten zählenden Treffer + 5 s gesetzt.

Nur Enden und Anfänge, die dadurch nach AUSSEN wandern — die Regel
verlängert nie einen Block über eine Kante hinaus, an der kein Marker
steht, und verkürzt nie.

## Was der Schattenlauf schreibt

Je Aufnahme mit menschlichem Label, ab dem Tag der Aktivierung, eine Zeile:
uuid, Zeitstempel, je Kante der Ist-Fehler und der Fehler, den die Regel
gesetzt hätte, plus ob die Regel diese Kante überhaupt angefasst hat.

Die Regel wird dabei **nicht angewandt**. Die Cutlist bleibt unverändert.

## Bedingung — festgelegt, bevor eine Zeile existiert

Ausgewertet wird erst bei **mindestens 40 angefassten Kanten** aus
**mindestens 20 verschiedenen Aufnahmen**. Vorher wird nicht
hineingeschaut — kein Zwischenstand, keine „Tendenz".

**ERFÜLLT, wenn alle drei gelten:**

1. Die Summe des Kantenfehlers über die angefassten Kanten sinkt um
   **≥ 25 %**.
2. **Mehr Kanten besser als schlechter**, im Verhältnis mindestens **2:1**.
3. Keine einzelne Kante wird um mehr als **30 s** schlechter.

Zusätzlich berichtet, nicht Bedingung: der Kantenfehler über ALLE Kanten
(auch die unangefassten), der Anteil Kanten ≤ 2 s vorher/nachher, und wie
viele Aufnahmen die Regel gar nicht berührt hat.

⚠️ Bedingung 1 ist bewusst als **relative** Größe auf den angefassten
Kanten formuliert. Bei O11 hatte ich eine absolute Schwelle aus O9
übernommen, wo ein globaler Regler jede Aufnahme traf — auf einen
spärlichen Eingriff übertragen war sie strukturell blind. Der Fehler wird
hier nicht wiederholt; dafür ist die Hürde mit 25 % und 2:1 deutlich höher
als alles, was O9–O12 verlangt haben.

## Vorhersage

Auf den bisherigen (verbrauchten) Sätzen lag der Effekt bei den angefassten
Aufnahmen bei +0.020 und +0.031 IoU, mit 7:3 und 5:1. Übersetzt auf
Kantensekunden erwarte ich **ERFÜLLT**.

Ich habe bei O9, O10, O11 und O12 jeweils „besteht" vorhergesagt und lag
jedes Mal falsch. Das ist protokolliert und gehört zur Einordnung dieser
fünften Vorhersage: meine Zuversicht ist bei dieser Idee nachweislich zu
hoch kalibriert.

## Was jeder Ausgang bedeutet — jetzt entschieden

**ERFÜLLT** → die Regel wird angewandt (hinter einem Schalter, erst nach
einer bestätigenden Nacht scharf), und OCR läuft in Produktion mit.

**NICHT ERFÜLLT** → die Spur ist zu Ende, endgültig. Fünf Anläufe sind
genug; der Befund §3aa (der Backbone kann die Einblendung nicht lesen)
bleibt gültig, und die Konsequenz ist dann die teure: das Merkmal gehört in
den KOPF (eigene Spalte, Neu-Extraktion des Korpus), nicht dahinter. Kein
sechster nachgelagerter Regelversuch.

⚠️ Kein Ausgang rechtfertigt, den Golden-Boden zu senken (L1) oder Labels
anzufassen (L2). Insbesondere werden die Kanten, an denen die Regel
verliert, **nicht** per Frame-Review geprüft, um sie als Label-Fehler
umzudeuten — das war bei O10 der erste Blick auf einen unbekannten
Fehlermodus und ist es kein zweites Mal.
