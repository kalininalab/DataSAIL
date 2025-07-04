import datetime
import doctest
import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("./.."))
sys.path.insert(0, os.path.abspath("./."))

import datasail
from datasail.version import __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "nbsphinx",
    "nbsphinx_link",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_parser",
]

suppress_warnings = ["config.cache"]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["build", "_build", "_templates", "**.ipynb_checkpoints"]

source_suffix = ".rst"
master_doc = "index"

author = "Roman Joeres"
project = "DataSAIL"
copyright = f"{datetime.datetime.now().year}, {author}"

release = __version__

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "navigation_depth": 2,
}
html_context = {}
html_logo = "imgs/DataSAIL_Logo.png"
html_show_sourcelink = True

rst_context = {"DataSAIL": datasail}
mathjax3_config = {'chtml': {'displayAlign': 'left'}}

add_module_names = False
fail_on_warning = True
