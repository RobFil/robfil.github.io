# Japanischtrainer Skill

Professionelle Skill-Struktur fuer einen lokalen Japanischtrainer. Rohmaterial bleibt lokal, der Commit enthaelt nur Skill-Code, Schema, Skripte, Tests und leere Arbeitsordner.

## Ziel

- Japanisch-PDFs und Textquellen lokal ablegen.
- Inhalte extrahieren, pruefen und als akzeptierten Lernkontext speichern.
- Maschinenlesbare Daten und menschenlesbare Review-Dateien getrennt halten.
- PDFs, OCR-Zwischenstaende, Debug-Ausgaben und generierte Extrakte nicht unnoetig versionieren.

## Struktur

```text
files/language/teacher/
  SKILL.md
  README.md
  agents/openai.yaml
  input/
    pdfs/          lokale Roh-PDFs, nicht im Git
    text/          lokale OCR- oder Text-Exporte, nicht im Git
  output/
    machine/       generierte JSON/JSONL-Daten, nicht im Git
    human/         Review-Markdown, nicht im Git
    debug/         OCR- und Extraktionsdiagnose, nicht im Git
  references/
    data-schema.md
    jlpt/          kleine optionale JLPT-Referenzdaten
  scripts/
  tests/
```

## Git-Policy

Eingecheckt werden:

- Skill-Anweisungen und Metadaten
- Skripte und Tests
- `references/data-schema.md`
- kleine, lizenzierte Referenzdaten
- `.gitkeep` fuer leere Arbeitsordner

Nicht eingecheckt werden:

- PDFs
- lokale Textimporte
- generierte `output/`-Inhalte
- OCR- und Debug-Artefakte
- Python-Caches

## Typischer Ablauf

PDFs lokal nach `input/pdfs/` legen und dann die Extraktion bzw. Verwaltung ueber die Skripte im Ordner `scripts/` ausfuehren. Akzeptierter Kontext wird lokal unter `output/` erzeugt und kann jederzeit neu aufgebaut werden.

```powershell
.venv\Scripts\python.exe files\language\teacher\scripts\manage_extractions.py --help
```

Das Datenmodell steht in `references/data-schema.md`.
