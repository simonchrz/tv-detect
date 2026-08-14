import Foundation
import Vision
import AppKit

// Liest jedes uebergebene Bild und schreibt eine Zeile
//   <pfad>\t<erkannter text, mit | getrennt>
// Vision laeuft lokal, kein Netz.
for pfad in CommandLine.arguments.dropFirst() {
    guard let bild = NSImage(contentsOfFile: pfad),
          let cg = bild.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("\(pfad)\t"); continue
    }
    let anfrage = VNRecognizeTextRequest()
    anfrage.recognitionLevel = .accurate
    anfrage.recognitionLanguages = ["de-DE", "en-US"]
    anfrage.usesLanguageCorrection = false
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
