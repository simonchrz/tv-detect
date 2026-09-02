import Foundation
import Vision
import AppKit

// Liest jedes uebergebene Bild und schreibt eine Zeile
//   <pfad>\t<erkannter text, mit | getrennt>
// Vision laeuft lokal, kein Netz.
//
// ⚠️ VIER OPTIMIERUNGEN GEMESSEN, DREI VERWORFEN (2026-09-02). Anlass: OCR
// traegt 78 % der Kosten der Erhebung (je 180-s-Fenster: ffmpeg 1,27 s gegen
// Vision 4,47 s). Wer hier etwas verbessern will, findet unten, was schon
// geprueft ist — bitte nicht noch einmal.
//
// 1. NEBENLAEUFIGKEIT — kein Gewinn, verworfen.
//    Die Schleife lief seriell, also lag der Verdacht nahe. Gemessen:
//    4,13 s seriell gegen 4,20 s mit sechs parallelen Auftraegen, Ausgabe
//    zeilenidentisch. Der Grund steht in der Auslastung: user+sys 1,93 s bei
//    real 4,17 s sind 0,5 Kerne. Der Prozess RECHNET nicht, er WARTET —
//    Vision schiebt die Erkennung auf die Neural Engine, und die ist der
//    Engpass. CPU-Parallelitaet kann daran nichts aendern.
//
// 2. .fast STATT .accurate — 7,7x schneller, unbrauchbar.
//    18,97 s -> 2,45 s ueber 495 Frames, aber 56 der 83 Treffer verloren:
//    die Stufe verstuemmelt genau die Woerter, an denen die Regel haengt
//    ("WERBUNG" -> "WER", "GRATIS" -> "G1411S").
//
// 3. .fast ALS VORFILTER (accurate nur auf Frames mit Text) — 1 %.
//    Kein einziger echter Treffer wurde von .fast uebersehen, aber .fast
//    findet in 86 % der Frames irgendeinen Text: das Senderlogo steht immer
//    im Bild. Mit einer Laengenschwelle von 5 Zeichen sind es 7 % Ersparnis
//    ohne Verlust — zu wenig fuer die Komplexitaet.
//
// 4. ZUSCHNITT — siehe internal/signals/ocr.go, ebenfalls verworfen:
//    die Marker sitzen in ZWEI Bildregionen.
//
// Was bleibt: die Kosten sind der Preis dafuer, 90 volle Frames je Kante
// genau zu lesen. Guenstiger wird es nur ueber WENIGER Frames — und die
// Abtastdichte von 2 s ist in der O13-Registrierung festgeschrieben, sie zu
// aendern wuerde das Experiment beschaedigen, fuer das die Erhebung laeuft.
//
// 5. minimumTextHeight — DAS FUNKTIONIERT, seit 2026-09-02 aktiv (0.08).
//    Vision durchsucht sonst das Bild bis hinunter zu winziger Schrift. Die
//    Marker, auf die es hier ankommt, sind grosse Einblendungen — die
//    Feinsuche ist also reine Verschwendung. Gemessen:
//
//        Aufnahme          minh=0     minh=0.08     Treffer
//        kabel-eins       20,1 s      12,9 s        83 -> 80
//        GZSZ (Fenster)    4,3 s       2,7 s        14 -> 14
//
//    Die drei fehlenden Bilder liegen INNERHALB ihrer Ereignisse: der
//    aeusserste Treffer je Kantenfenster ist bei jeder Stufe identisch, und
//    genau der bestimmt, wohin die Regel die Kante setzt. Kein Ereignis geht
//    verloren.
//
//    Abstand nach oben ist reichlich: bis minh=0.20 aendert sich nichts mehr
//    (getestet 0.04/0.05/0.07/0.08/0.10/0.14/0.20). 0.08 liegt also weit im
//    flachen Bereich, nicht an der Kante. TVOCR_MINHEIGHT=0 schaltet ab.
//
// TVOCR_LEVEL=fast bleibt als Messwerkzeug erhalten, NICHT fuer den Betrieb.
let stufe: VNRequestTextRecognitionLevel =
    (ProcessInfo.processInfo.environment["TVOCR_LEVEL"] == "fast") ? .fast : .accurate

// Mindest-Texthoehe als Anteil der Bildhoehe. 0 = Vorgabe von Vision (sucht
// bis zur kleinsten Schrift).
let minHoehe: Float = {
    if let s = ProcessInfo.processInfo.environment["TVOCR_MINHEIGHT"],
       let v = Float(s) { return v }
    return 0.08
}()

for pfad in CommandLine.arguments.dropFirst() {
    guard let bild = NSImage(contentsOfFile: pfad),
          let cg = bild.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("\(pfad)\t"); continue
    }
    let anfrage = VNRecognizeTextRequest()
    anfrage.recognitionLevel = stufe
    anfrage.recognitionLanguages = ["de-DE", "en-US"]
    anfrage.usesLanguageCorrection = false
    if minHoehe > 0 { anfrage.minimumTextHeight = minHoehe }
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    var texte: [String] = []
    do {
        try handler.perform([anfrage])
        for r in (anfrage.results ?? []) {
            if let t = r.topCandidates(1).first, t.confidence > 0.3 {
                texte.append(t.string)
            }
        }
    } catch { }
    print("\(pfad)\t\(texte.joined(separator: " | "))")
}
