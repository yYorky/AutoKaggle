"""Fetch and clean Kaggle competition page content."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional
from urllib.request import Request, urlopen


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._skip_stack and self._skip_stack[-1] == tag:
            self._skip_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        cleaned = data.strip()
        if cleaned:
            self._chunks.append(cleaned)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def fetch_competition_page_text(competition_url: str, timeout: int = 10) -> str | None:
    """Fetch competition page HTML and return a cleaned text excerpt."""
    try:
        request = Request(competition_url, headers={"User-Agent": "AutoKaggle/1.0"})
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL provided by user
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="ignore")
    except Exception:
        return None
    if not html:
        return None
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned or None
