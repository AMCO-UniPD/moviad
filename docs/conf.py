"""Documentation configuration; builds without importing ML dependencies."""
from pathlib import Path
import tomllib

metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
project = "MoViAD"
author = "Manuel Barusco, Francesco Borsatti, Youssef Ben Khalifa, Davide Dalle Pezze"
copyright = "2026, " + author
release = metadata["project"]["version"]
extensions = ["sphinx.ext.napoleon", "sphinx.ext.viewcode"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["moviad.css"]
html_title = "MoViAD · Visual anomaly detection"
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
html_context = {"display_github": False}
