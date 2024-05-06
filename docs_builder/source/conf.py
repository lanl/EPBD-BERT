import sys
import os

sys.path.append(os.path.abspath(os.path.join("../../")))
from epbd_bert.version import __version__

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EPBD-BERT"
copyright = "2023, LANL"
author = "Anowarul Kabir, Manish Bhattarai, Kim Rasmussen, Amarda Shehu, Alan Bishop, Boian Alexandrov and Anny Usheva"
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autoclass_content = "both"
templates_path = ["_templates"]
exclude_patterns = [
    "version",
    "AutoTokenizer",
    "AutoModel",
    "AutoModelForSequenceClassification",
    "utility",
    "train_lightning",
    "test",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"  # alabaster, furo, sphinx_book_theme, sphinx_rtd_theme
html_static_path = ["_static"]

pygments_style = "sphinx"

# add the folling line in the index.rst inside 'toctree'
# modules/modules
