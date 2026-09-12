"""Deterministic Japanese-language rules for ``parse_pdfs.py``.

This module contains no AI calls. Keeping these heuristics separate makes the
PDF extraction script usable for documents in any language.
"""

from __future__ import annotations

import re


THEME_PATTERNS = {
    "vocabulary": [
        r"\b(vocabulary|vocab|wortschatz)\b|単語|語彙",
        r"[一-龯々ぁ-んァ-ヶー]+\s*[-=]\s*[A-Za-zÄÖÜäöüß]",
    ],
    "kanji": [r"\bkanji\b|漢字", r"kun-?yomi|on-?yomi|訓読み|音読み"],
    "expressions": [
        r"\b(expression|phrase|redewendung)\b|表現|言い回し|言い換え|フレーズ|定型句",
        r"敬語|尊敬語|謙譲語|丁寧語",
    ],
    "grammar": [
        r"\b(grammar|grammatik)\b|文法",
        r"ます|です|て-?form|ない形",
    ],
    "dialogues": [r"\b(dialogue|dialog)\b|会話", r"^[ABＡＢ][：:]", r"さん[：:]"],
    "readings": [r"\b(reading|lesetext)\b|読解|読み物"],
    "exercises": [r"\b(exercise|uebung|übung|aufgabe)\b|問題|練習"],
}

JAPANESE_RE = re.compile(r"[一-龯々ぁ-んァ-ヶー]")
LATIN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]")


def classify_theme(text: str) -> tuple[str, str]:
    """Classify a Japanese-learning text page using transparent regex rules."""
    compact = text[:3000]
    for theme, patterns in THEME_PATTERNS.items():
        if any(
            re.search(pattern, compact, flags=re.IGNORECASE | re.MULTILINE)
            for pattern in patterns
        ):
            return theme, "medium"
    return "uncategorized", "low"


def detect_language_hint(text: str) -> str:
    """Return ``ja``, ``mixed``, or ``unknown`` from visible script ranges."""
    has_japanese = bool(JAPANESE_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))
    if has_japanese and has_latin:
        return "mixed"
    if has_japanese:
        return "ja"
    return "unknown"
