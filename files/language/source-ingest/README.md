# Japanese Source Ingest Skill

Skill-Struktur fuer das lokale Aufbereiten von Japanisch-Lernmaterialien. Dieser Skill ist fuer Datenaufbau, PDF/OCR-Verarbeitung, Review und Kontextgenerierung zustaendig; das eigentliche Unterrichten liegt im separaten `teacher`-Skill.

## Ziel

- Japanisch-PDFs und Textquellen lokal ablegen.
- Inhalte extrahieren, pruefen und als akzeptierten Lernkontext speichern.
- Maschinenlesbare Daten und menschenlesbare Review-Dateien getrennt halten.
- PDFs, OCR-Zwischenstaende, Debug-Ausgaben und generierte Extrakte nicht unnoetig versionieren.

## Struktur

```text
files/language/source-ingest/
  SKILL.md
  README.md
  agents/openai.yaml
  input/
    pdfs/          lokale Roh-PDFs, nicht im Git
    text/          lokale OCR- oder Text-Exporte, nicht im Git
  output/
    machine/       generierte JSON/JSONL-Daten, nicht im Git
    debug/         optionale OCR- und Extraktionsdiagnose, nicht im Git
  references/
    data-schema.md
    jlpt/          kleine optionale JLPT-Referenzdaten
  scripts/
  tests/
```

## Git-Policy

Eingecheckt werden Skill-Anweisungen, Metadaten, Skripte, Tests, Schema, kleine lizenzierte Referenzdaten und `.gitkeep`-Platzhalter. Nicht eingecheckt werden PDFs, lokale Textimporte, generierte `output/`-Inhalte, OCR-/Debug-Artefakte und Python-Caches.

## Typischer Ablauf

```powershell
.venv\Scripts\python.exe files\language\source-ingest\scripts\manage_extractions.py --help
```

Akzeptierter Kontext wird lokal nur unter `output/machine/context.jsonl` erzeugt. Der `teacher`-Skill nutzt diese Datei als Lernbasis. Fuer eine Fehleranalyse kann `extract-new` mit `--debug-review` zusaetzliche lokale Markdown-Dateien und Seitenbilder erzeugen.
