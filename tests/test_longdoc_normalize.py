"""Testes para normalize_longdoc_page."""

from schemas import normalize_longdoc_page


def test_strips_markdown_fence():
    raw = "```markdown\n# Title\n\nSome text.\n```"
    assert normalize_longdoc_page(raw) == "# Title\n\nSome text."


def test_strips_plain_fence():
    raw = "```\n# Title\n```"
    assert normalize_longdoc_page(raw) == "# Title"


def test_converts_literal_newlines():
    raw = "Line 1\\nLine 2"
    assert normalize_longdoc_page(raw) == "Line 1\nLine 2"


def test_strips_unclosed_fence():
    raw = "```markdown\n# Title\n\nSome text."
    assert normalize_longdoc_page(raw) == "# Title\n\nSome text."


def test_clean_text_unchanged():
    raw = "# Heading\n\nParagraph."
    assert normalize_longdoc_page(raw) == raw


def test_strips_incomplete_figure_blockquote():
    raw = "# Title\n\nSome text.\n\n> **Figura"
    assert normalize_longdoc_page(raw) == "# Title\n\nSome text."


def test_strips_partial_bibliography_footnote():
    raw = (
        "# Section\n\nParagraph with [58] inline.\n\n"
        '---\n\n> **Nota de rodapé:** [58] Author et al., "Wild'
    )
    assert normalize_longdoc_page(raw) == "# Section\n\nParagraph with [58] inline.\n\n---"
