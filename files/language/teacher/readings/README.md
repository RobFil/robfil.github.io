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

## Post-Schema

```yaml
---
layout: post
title: "Japanischer Titel"
date: 2026-09-13
categories: [reading]
topic: Technik
level: N2
grammar:
  - "〜た上で"
  - "〜可能性があります"
summary_de: "Kurze deutsche Zusammenfassung."
generated: true
fictional: true
---
```

Unterhalb des japanischen Texts folgt eine kurze Lernnotiz mit den verwendeten
Grammatikmustern. Die Front Matter ist zugleich die maschinenlesbare Zuordnung
für spätere Auswahl- und Wiederholungslogik.
