"""Catch import-blocking syntax errors without loading optional ML dependencies."""
from pathlib import Path


def test_all_library_sources_compile():
    root = Path(__file__).resolve().parents[2] / "src"
    files = sorted(root.rglob("*.py"))
    assert files, "The library source directory must not be empty"
    for source in files:
        compile(source.read_text(), str(source), "exec")
