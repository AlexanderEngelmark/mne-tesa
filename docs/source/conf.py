# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "mne_tesa"
copyright = "2025, Alexander Engelmark"
author = "Alexander Engelmark"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "mne": ("https://mne.tools/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]

html_theme_options = {
    "logo": {
        "text": "MNE-TESA",
    },
    "icon_links": [
        {
            "name": "Codeberg",
            "url": "https://codeberg.org/AlexanderEngelmark/mne-tesa",
            "icon": "fa-brands fa-codeberg",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/AlexanderEngelmark/mne-tesa",
            "icon": "fa-brands fa-github",
        },
    ],
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "secondary_sidebar_items": ["page-toc"],
    "back_to_top_button": True,
}

html_show_sourcelink = False
html_show_sphinx = False
html_copy_source = False
