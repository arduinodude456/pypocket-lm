# PyPocket LM

**PyPocket LM** ist ein kleiner, kostenloser Forschungsprototyp für Python-Codegenerierung direkt im Browser. Die aktuelle Version kombiniert eine Python-aware Code-Fortsetzung mit einem trainierten Prompt-/Intent-Layer:

```text
Python-Beispiele + natürliche Prompts → strukturierte Tokens und Prompt-Wortschatz → kleines JSON-Modell → Browser-Inferenz
```

Die GitHub-Action trainiert ausschließlich mit Python-Standardbibliothek. Das exportierte Modell liegt als JSON vor und kann von einer GitHub-Pages-Seite geladen werden. Die Browser-Inferenz benötigt keinen Server, keinen API-Key und keinen Manus-LLM-Aufruf.

## Was dieser Prototyp kann

- Python-Schlüsselwörter und häufige Syntaxelemente als kompakte Spezialtokens behandeln.
- Ein kleines, transparentes 3-Gramm-Modell aus Python-Beispielen trainieren.
- Natürliche deutsche und englische Prompts über Wortmuster einem Python-Intent zuordnen.
- Parametrisierte Grundaufgaben wie Hallo Welt, Begrüßungsfunktionen, Summen, Schleifen und Fibonacci erzeugen.
- JSON-Gewichte und Vokabular exportieren.
- Aus einem Prompt eine kurze Tokenfolge vervollständigen.
- Das Modell direkt im Browser laden und ausführen.
- Neue persönliche Trainingsbeispiele über `data/examples.py` ergänzen.
- Eine experimentelle kompakte Python-IR erzeugen (`SET`, `ADD`, `MUL`, `PRINT`, `FOR_RANGE` usw.).
- IR mit einem autoregressiven Next-Token-n-Gramm-Modell erzeugen und durch einen deterministischen Compiler in Python übersetzen.
- Trainings- und unbekannte Kompositionsaufgaben getrennt evaluieren (`web/metrics.json`).

## Was er noch nicht kann

Dies ist **kein ChatGPT-ähnliches Large Language Model**. Der Prompt-Layer ist eine kleine, trainierte Intent-/Template-Komponente und kein vollständiges semantisches Sprachmodell. Unbekannte Prompts fallen auf die Code-Fortsetzung zurück; komplexe Anforderungen, vollständige Bibliothekskenntnis und zuverlässig produktionsreifer Code sind noch nicht garantiert. Die Architektur ist absichtlich klein, erklärbar und kostenlos, damit Training und Inferenz auf begrenzter Hardware möglich bleiben.

Die IR-Erweiterung ist der ehrlichere Generalisierungsversuch: Der Prompt wird in Konzept-Tokens zerlegt, das autoregressive Modell erzeugt anschließend IR Token für Token, und der Compiler erzeugt daraus Python. Im aktuellen reproduzierbaren Lauf lag die exakte Trefferquote auf den Trainingsaufgaben bei 100 %, auf unbekannten synthetischen Kombinationen bei 21,7 %. Gleichzeitig waren 100 % der erzeugten Validierungs-IRs und 100 % der daraus kompilierten Python-Ausgaben syntaktisch gültig. Das zeigt echte Komposition mit noch schwacher semantischer Genauigkeit – nicht „fertige Intelligenz“.

## IR-Training und Evaluation

```bash
python3 train_ir.py --output web/ir_model.json --metrics web/metrics.json
python3 -m unittest discover -s tests -v
```

Die Evaluation trennt Training von unbekannten Kombinationen und misst Cross-Entropy, exakte IR-Übereinstimmung, IR-Validität und AST-basierte Python-Syntax. Die Browser-Demo bietet dafür den Schalter **Creative IR**. Generierten Code niemals ungeprüft ausführen.

## Lokal starten

```bash
python3 train.py
python3 -m unittest discover -s tests -v
python3 -m http.server 8080 --directory web
```

Danach `http://localhost:8080` öffnen. Die Demo lädt `model.json` aus dem `web/`-Verzeichnis.

## Eigene Beispiele ergänzen

Bearbeite `data/examples.py` und füge Python-Code als String in `TRAINING_EXAMPLES` oder neue natürliche Prompt-/Intent-Beispiele in `PROMPT_EXAMPLES` ein. Die Erkennung basiert auf Wortmustern, nicht auf exaktem Prompt-Matching. Danach erneut trainieren:

```bash
python3 train.py --epochs 2
```

## GitHub Pages und Actions

Der Workflow `.github/workflows/train.yml` läuft bei Änderungen an Trainingsdaten oder manuell. Er trainiert das Modell, führt die Tests aus, kopiert den JSON-Artefakt nach `web/model.json` und veröffentlicht `web/` auf GitHub Pages.

Im Repository müssen unter **Settings → Pages** die GitHub-Actions-Quelle und einmalig Pages aktiviert werden. Für öffentliche Repositories fallen für diesen Standardbibliotheks-Workflow keine Manus-Credits an; GitHub-Kontingente und GitHub-Regeln gelten weiterhin.

## Sicherheit

Die Browser-Demo führt den generierten Text nicht aus. Generierten Python-Code immer prüfen, bevor er lokal ausgeführt wird.
