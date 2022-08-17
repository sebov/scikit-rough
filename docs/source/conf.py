# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "scikit-rough"
copyright = "2022, sebov"
author = "sebov"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []
extensions = [
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "myst_nb",
    # "sphinx_thebe",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

templates_path = ["_templates"]
# exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# https://bashtage.github.io/sphinx-material
html_theme = "sphinx_material"
html_static_path = ["_static"]


# defaults from jupyter-book
#
# language = "en"
# pygments_style = "sphinx"
# html_add_permalinks = "¶"
# html_sourcelink_suffix=""
# numfig = True
# panels_add_bootstrap_css = False
# suppress_warnings=["myst.domains"]
