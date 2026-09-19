# Lesetexte

Dieses Verzeichnis beschreibt den Veröffentlichungsweg für vom Teacher-Skill
erstellte Lesetexte. Die veröffentlichte, Yomitan-freundliche Quelle ist jeweils
eine Markdown-Datei in `_posts/reading/`.

## Anforderungen

- Der japanische Text ist neu erstellt und keine Abschrift aus Unterrichtsmaterial.
- Grammatikziele werden aus akzeptiertem Lernkontext oder aus einem ausdrücklich
  genannten Lernziel gewählt.
- Nachrichtenstil ist immer als Übungstext markiert, sofern keine aktuelle Quelle
  zitiert wird.
- Der eigentliche japanische Text bleibt zusammenhängend und ohne Furigana,
  Romaji oder eingebettete Übersetzungen.
- Jeder Text erzählt eine Sache: Ausgangslage, Entwicklung und Ergebnis gehören
  sichtbar zusammen. Grammatik ist ein Mittel der Erzählung, nicht deren Thema.
- Für N2 sind in der Regel 700-1.100 japanische Zeichen sinnvoll. Kürzere Texte
  gibt es nur auf ausdrücklichen Wunsch oder bei einem niedrigeren Niveau.
- Der Standard ist ein Paar aus zwei Texten mit insgesamt etwa 20 Minuten
  Lernzeit. Jeder Post hat ungefähr 10 Minuten Lernzeit und trägt das Feld
  `reading_time_minutes: 10`.
- Echte Nachrichten erhalten die ursprüngliche Quelle, deren Datum und das
  Abrufdatum. Die japanische Fassung ist stets eine eigene Zusammenfassung.

## Post-Schema

```yaml
---
layout: reading
title: "Japanischer Titel"
date: 2026-09-13
categories: [reading]
topic: Technik
level: N2
reading_time_minutes: 10
grammar:
  - "〜た上で"
  - "〜可能性があります"
source_selection: varied # guided | random | varied | active_batch | general_n2
learning_sources:
  - "20260607_Unit4_不具合・障害報告_教材.pdf"
summary_de: "Kurze deutsche Zusammenfassung."
generated: true
fictional: true
source_url: null
source_published_at: null
source_checked_at: null
---
```

Unterhalb des japanischen Texts folgt eine kurze Lernnotiz mit den verwendeten
Grammatikmustern. Die Front Matter ist zugleich die maschinenlesbare Zuordnung
für spätere Auswahl- und Wiederholungslogik. `learning_sources` enthält nur PDFs,
die die Grammatikwahl tatsächlich beeinflusst haben; bei allgemeinem Material
steht dort `general_n2`.
