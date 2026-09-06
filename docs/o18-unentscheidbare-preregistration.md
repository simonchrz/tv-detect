# O18 — Sollen auch die nicht entscheidbaren Aufnahmen ihr Privileg verlieren?
(Vorab-Registrierung)

**Geschrieben 2026-09-06, bevor ein einziger Fit mit `--herkunft-streng`
gerechnet wurde.** Bedingung, Paarzahl und Schwelle stehen fest, bevor die
erste Zahl existiert (R4).

## Woher die Frage kommt

O17 hat `has_user` von der **Existenz** einer Datei auf die **Marker**
darin umgestellt und ist seit heute im Nightly scharf. Damit sind drei
Zustände unterscheidbar:

| `mensch_belegt` | Lage | O17 |
|---|---|---|
| `True` | Marker sagen: Mensch | behält `has_user` |
| `False` | Marker sagen: auto-confirm / Fingerprint / Werkzeug | **verliert es** (92 korpusweit) |
| `None` | keine lesbare Quelle mehr — Aufnahme tot, nur Archiv-Eintrag | behält es (**294 im train-Eimer**) |

Die 294 sind die eigentliche Masse. Sie behalten heute das doppelte
Gewicht, die Ausnahme vom Hygiene-Veto, das deploy-blockende
Reviewed-Regression-Veto und die Ausnahme vom GT-Ausreißer-Wächter — auf
Grundlage von `which="merged"`, und genau dieses Kriterium hat sich als
untauglich erwiesen. Über ihre Herkunft ist damit **nichts** bekannt.

## Die Frage

Ist es besser, sie wie Maschinenlabels zu behandeln — oder ist die
Unwissenheit ein Grund, sie in Ruhe zu lassen?

## Warum die Antwort nicht offensichtlich ist

Beide Richtungen haben ein ernsthaftes Argument, und deshalb wird gemessen
statt entschieden:

* **Für Degradieren:** eine Aufnahme, deren Quelle seit Monaten weg ist,
  wurde in aller Regel nie reviewt — der Review-Anteil im Korpus liegt
  bei 59 % der lebenden, und die toten stammen überwiegend aus der Zeit,
  in der auto-confirm schon lief. Ein Privileg auf Verdacht ist kein
  Privileg, sondern ein Zufall.
* **Gegen Degradieren:** unter den 294 sind mit Sicherheit auch echte,
  mühsam von Hand reviewte Aufnahmen aus der Zeit vor auto-confirm. Ihnen
  das Gewicht zu nehmen wirft menschliche Arbeit weg — und zwar die
  älteste, also die aus der Zeit, als noch regelmäßig reviewt wurde.

⚠️ **Diese Frage ist NICHT durch Nachschauen zu klären.** Die Marker sind
mit der Quelle verschwunden; das Label-Backup spiegelt nur, was noch da
ist. Wer behauptet, die 294 seien „vermutlich maschinell", trifft eine
Annahme über 294 Aufnahmen und nennt sie eine Feststellung.

## Arme

* **`belegt`** — der ab heute geltende Zustand (`--herkunft-belegt`).
* **`streng`** — zusätzlich verlieren die `None`-Fälle `has_user`
  (`--herkunft-belegt --herkunft-streng`).

Die Grundlinie ist bewusst `belegt` und nicht der Stand von gestern: O17
ist entschieden und eingebaut, und zwei Änderungen in einer Zahl wären
nicht auseinanderzuhalten.

## Bedingung

```regel
{
  "id": "O18",
  "frage": "Sollen auch die nicht entscheidbaren Aufnahmen has_user verlieren?",
  "serie_art": "tagesserie",
  "serie_ab": "20260906",
  "naechte": 5,
  "arme": {"mit": "mlp32-cwtmpwm-streng", "ohne": "mlp32-cwtmpwm-belegt"},
  "delta": "streng minus belegt, auf golden_median, beide Arme gleicher Seed",
  "gueltige_nacht": {
    "set_hash": "c8727e8266a8",
    "decoder": "--decoder hsmm --hsmm-dur-w 15",
    "golden_n": 38
  },
  "bedingungen": {
    "median_mindestens": 0.010,
    "positive_naechte_mindestens": 4
  }
}
```

Schwelle und Paarzahl wie bei O17: der Rauschboden (§2) ist 0.008 Std /
0.023 Spanne über 5 Seeds, alles unter 0.010 wäre nach R2 kein Ergebnis.

**Vorhersage, vor der ersten Zahl.** Anders als bei O17 erwarte ich hier
**keine** klare Richtung. O17 betraf 92 Aufnahmen und bewegte den Median
um +0.0032; O18 betrifft mit 294 gut das Dreifache, aber das Vorzeichen
ist offen — es hängt daran, wie viele der 294 echte Reviews sind, und
genau das weiß niemand. Wenn ich raten müsste: ein Ergebnis im Rauschen,
also erneut „nicht erfüllt".

**Konsequenz bei Erfüllen:** `--herkunft-streng` wird scharfgeschaltet.

**Konsequenz bei Verfehlen (jetzt festgelegt):** `--herkunft-streng`
bleibt **AUS**, und zwar dauerhaft, bis eine neue Tatsache vorliegt — nicht
„bis es sich besser anfühlt". Begründung, und sie unterscheidet O18 von
O17: bei O17 war die Korrektur auch ohne Messgewinn richtig, weil `which`
eine nachweislich **falsche Tatsachenbehauptung** war. Hier gibt es keine
falsche Behauptung zu korrigieren — nur Unwissenheit. Ohne belegten Nutzen
ist die konservative Wahl, menschliche Arbeit nicht auf Verdacht
wegzuwerfen.

⚠️ **Ein deutlich NEGATIVER Median wäre das interessantere Ergebnis.** Er
hiesse: unter den 294 steckt echte menschliche Arbeit, die das Modell
braucht. Das ist keine Bedingung dieser Registrierung — aber es wird
berichtet, nicht verschwiegen, falls es eintritt.

## Was diese Frage NICHT beantwortet

Ob der Maßstab selbst zu reparieren ist (golden 22/38 nachweislich
menschlich, test 7/102, versiegelt 0/37). Das bleibt die größere offene
Sache und ist von beiden Armen unberührt.
