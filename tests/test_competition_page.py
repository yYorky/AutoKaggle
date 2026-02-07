from __future__ import annotations

from autokaggle.competition_page import _HTMLTextExtractor


def test_html_text_extractor_strips_script_and_style() -> None:
    html = """
    <html>
      <head>
        <style>body { color: red; }</style>
        <script>console.log("skip");</script>
      </head>
      <body>
        <h1>Competition Title</h1>
        <p>Evaluation metric: RMSE</p>
      </body>
    </html>
    """
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    assert "Competition Title" in text
    assert "Evaluation metric: RMSE" in text
    assert "console.log" not in text
