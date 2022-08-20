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

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.apidoc",
    # "sphinx_thebe",
]

# sphinxcontrib-apidoc settings - https://github.com/sphinx-contrib/apidoc
apidoc_module_dir = "../../src/skrough"
apidoc_output_dir = "reference"
apidoc_separate_modules = True

templates_path = ["_templates"]
# exclude_patterns = []
# suppress_warnings=["myst.domains"]
# pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"

# https://bashtage.github.io/sphinx-material
html_theme = "sphinx_material"

html_static_path = ["_static"]

if html_theme == "sphinx_material":
    # Material theme options (see theme.conf for more information)
    html_theme_options = {
        # Set the name of the project to appear in the navigation.
        # 'nav_title': 'Project Name',
        # Set you GA account ID to enable tracking
        # 'google_analytics_account': 'UA-XXXXX',
        # Specify a base_url used to generate sitemap.xml. If not
        # specified, then no sitemap will be built.
        "base_url": "https://project.github.io/project",
        # Set the color and the accent color
        # "color_primary": "grey",
        "color_accent": "indigo",
        # Set the repo location to get a badge with stats
        "repo_url": "https://github.com/sebov/scikit-rough",
        "repo_name": "scikit-rough",
        "repo_type": "github",
        # Visible levels of the global TOC; -1 means unlimited
        "globaltoc_depth": 3,
        # If False, expand all TOC entries
        "globaltoc_collapse": True,
        # If True, show hidden TOC entries
        "globaltoc_includehidden": True,
        # "html_minify": True,
        # "css_minify": True,
    }

    html_sidebars = {
        "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
    }

# -- Options for internationalization ----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-internationalization

language = "en"
