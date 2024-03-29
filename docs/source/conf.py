import os

from git.repo import Repo

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
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "sphinx_thebe",
]

templates_path = ["_templates"]
# exclude_patterns = []
# suppress_warnings=["myst.domains"]

# pygments_style = "default"
# pygments_style = "emacs"
pygments_style = "friendly"
# pygments_style = "friendly_grayscale"
# pygments_style = "colorful"
# pygments_style = "autumn"
# pygments_style = "murphy"
# pygments_style = "manni"
# pygments_style = "material"
# pygments_style = "monokai"
# pygments_style = "perldoc"
# pygments_style = "pastie"
# pygments_style = "borland"
# pygments_style = "trac"
# pygments_style = "native"
# pygments_style = "fruity"
# pygments_style = "bw"
# pygments_style = "vim"
# pygments_style = "vs"
# pygments_style = "tango"
# pygments_style = "rrt"
# pygments_style = "xcode"
# pygments_style = "igor"
# pygments_style = "paraiso-light"
# pygments_style = "paraiso-dark"
# pygments_style = "lovelace"
# pygments_style = "algol"
# pygments_style = "algol_nu"
# pygments_style = "arduino"
# pygments_style = "rainbow_dash"
# pygments_style = "abap"
# pygments_style = "solarized-dark"
# pygments_style = "solarized-light"
# pygments_style = "sas"
# pygments_style = "staroffice"
# pygments_style = "stata"
# pygments_style = "stata-light"
# pygments_style = "stata-dark"
# pygments_style = "inkpot"
# pygments_style = "zenburn"
# pygments_style = "gruvbox-dark"
# pygments_style = "gruvbox-light"
# pygments_style = "dracula"
# pygments_style = "one-dark"
# pygments_style = "lilypond"
# pygments_style = "nord"
# pygments_style = "nord-darker"
# pygments_style = "github-dark"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"

# https://bashtage.github.io/sphinx-material
# html_theme = "sphinx_material"

# https://pradyunsg.me/furo/
# html_theme = "furo"

# https://jbms.github.io/sphinx-immaterial/
# html_theme = "sphinx_immaterial"

# https://sphinx-book-theme.readthedocs.io/
html_theme = "sphinx_book_theme"

html_static_path = ["_static"]

# html_logo = "figures/rough_white.png"
html_logo = "figures/rough.png"

if html_theme == "sphinx_material":
    # need to be after autodoc or napoleon
    extensions.extend(
        [
            "sphinx_autodoc_typehints",
        ]
    )

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
        # "color_accent": "grey",
        # Set the repo location to get a badge with stats
        "repo_url": "https://github.com/sebov/scikit-rough",
        "repo_name": "scikit-rough",
        "repo_type": "github",
        "nav_title": "Scikit-rough Documentation",
        # Visible levels of the global TOC; -1 means unlimited
        "globaltoc_depth": 5,
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

if html_theme == "sphinx_immaterial":
    extensions.extend(
        [
            "sphinx_immaterial",
            # "sphinx_immaterial.apidoc.python.apigen",
        ]
    )

    html_theme_options = {
        "repo_url": "https://github.com/sebov/scikit-rough",
        "repo_name": "scikit-rough",
        "repo_type": "github",
        "icon": {
            "repo": "fontawesome/brands/github",
        },
    }

if html_theme == "sphinx_book_theme":
    if os.getenv("READTHEDOCS"):
        repository_branch = os.getenv("READTHEDOCS_VERSION_NAME")
        if repository_branch in ["latest", "stable"]:
            repository_branch = "main"
    else:
        repo = Repo(search_parent_directories=True)
        repository_branch = repo.active_branch.name
    html_theme_options = {
        "repository_url": "https://github.com/sebov/scikit-rough",
        "repository_branch": repository_branch,
        "use_repository_button": True,
        "launch_buttons": {
            "binderhub_url": "https://mybinder.org",
            "notebook_interface": "jupyterlab",
        },
    }

# -- Options for internationalization ----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-internationalization

language = "en"


# -- Extensions configuration ------------------------------------------------


# -- Settings for intersphinx ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# -- Settings for napoleon docs ----------------------------------------------
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/sphinxcontrib.napoleon.html
napoleon_use_rtype = False
