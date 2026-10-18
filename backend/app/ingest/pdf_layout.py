"""Small layout helpers for parsing port PDFs that have no ruling lines.

Indian port berthing programmes are laid out as fixed-column tables where a
single logical record occupies two visual lines. Plain text extraction
interleaves those lines, so anything that needs the working-vessel section has
to work from word coordinates instead.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    top: float


@dataclass
class Line:
    top: float
    words: list[Word]

    def in_span(self, x_min: float, x_max: float) -> list[Word]:
        """Words whose left edge falls inside a column span."""
        return [w for w in self.words if x_min <= w.x0 < x_max]

    def text_in_span(self, x_min: float, x_max: float) -> str:
        return " ".join(w.text for w in self.in_span(x_min, x_max)).strip()

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def left_x(self) -> float:
        return min((w.x0 for w in self.words), default=1e9)


def cluster_lines(raw_words: list[dict], tolerance: float = 3.5) -> list[Line]:
    """Group extracted words into visual lines by their vertical position.

    ``tolerance`` is the maximum drift in points between two words considered
    to sit on the same line. Port PDFs render superscript-ish digits slightly
    off the baseline, so a flat tolerance beats exact matching.
    """
    words = [
        Word(text=w["text"], x0=float(w["x0"]), x1=float(w["x1"]), top=float(w["top"]))
        for w in raw_words
        if w.get("text", "").strip()
    ]
    words.sort(key=lambda w: (w.top, w.x0))

    lines: list[Line] = []
    for word in words:
        if lines and abs(word.top - lines[-1].top) <= tolerance:
            lines[-1].words.append(word)
            # Track the running mean so drift does not accumulate.
            lines[-1].top = sum(w.top for w in lines[-1].words) / len(lines[-1].words)
        else:
            lines.append(Line(top=word.top, words=[word]))

    for line in lines:
        line.words.sort(key=lambda w: w.x0)
    return lines
