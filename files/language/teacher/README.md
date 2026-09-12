# Japanese Teacher Skill

Schlanker Skill fuer das eigentliche Japanischtraining. Dieser Skill unterrichtet, korrigiert, fragt ab und erzeugt neue Uebungen. Datenaufbau, PDF-Verarbeitung und Review gehoeren in den separaten `source-ingest`-Skill.

## Aufgabe

- Japanisch durch Dialog, Rollenspiel, Tests und Drills trainieren.
- Akzeptierten lokalen Kontext aus `../source-ingest/output/` nutzen, wenn vorhanden.
- Frische Uebungen erzeugen, statt Quellen wortgleich zu kopieren.
- Fehler knapp korrigieren und direkt weiterueben.

## Abgrenzung

Dieser Skill fuehrt keine PDF-Extraktion, OCR, Source-Review-Verwaltung oder Kontext-Rebuilds aus. Dafuer ist `files/language/source-ingest` zustaendig.
