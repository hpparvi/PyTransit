# Configuration file for the Sphinx documentation builder.
#
# For the full list of options see
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath('../..'))
# The figure helpers the plot directive imports live next to the documentation sources.
sys.path.insert(0, os.path.abspath('.'))

from pytransit import __version__

# -- Project information -----------------------------------------------------

master_doc = 'index'
project = 'PyTransit'
copyright = f'2010-{date.today().year}, Hannu Parviainen'
author = 'Hannu Parviainen'
release = __version__
version = '.'.join(__version__.split('.')[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx_design',
    'matplotlib.sphinxext.plot_directive',
    'nbsphinx',
]

templates_path = ['_templates']

exclude_patterns = [
    '**.ipynb_checkpoints',
    '_build',
    'Thumbs.db',
    '.DS_Store',
    # Notebooks that document the log posterior functions, which are not part of this
    # documentation set yet.
    'notebooks/examples/**',
    # Scratch notebooks that are not part of the documentation.
    '**/Untitled*.ipynb',
    # Example notebooks that no longer execute against the current API. They are kept in the
    # source tree but left out of the build until they are updated; see the note in the docs
    # README. Re-add them to the model pages' toctrees once they run.
    'notebooks/models/eclipse/*.ipynb',
    'notebooks/models/chromosphere/*.ipynb',
    'notebooks/models/uniform/*.ipynb',
    'notebooks/models/qpower2/*.ipynb',
    'notebooks/models/tsmodel/*.ipynb',
    'notebooks/models/quadratic/example_quadratic_opencl_model.ipynb',
    'notebooks/models/roadrunner/roadrunner_model_example_3.ipynb',
    'notebooks/models/roadrunner/roadrunner_model_theory.ipynb',
]

# Sort the autodoc members in the order they appear in the source rather than alphabetically:
# the model classes are written so that the initialiser, the data setup, and the evaluation
# follow the order in which they are called.
autodoc_member_order = 'bysource'

# The models are annotated with broad `Union[float, ndarray]`-style type hints that make the
# rendered signatures unreadable, and the docstrings describe the accepted types anyway.
autodoc_typehints = 'none'

autodoc_default_options = {
    'show-inheritance': True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

# -- matplotlib plot directive -----------------------------------------------

# The documentation figures are drawn from source at build time rather than stored, so a figure
# whose code stops working breaks the build instead of quietly going stale.
plot_formats = ['svg']
plot_html_show_source_link = False
plot_html_show_formats = False
plot_include_source = False

# -- nbsphinx ----------------------------------------------------------------

# Execute the example notebooks on every build so that their output always matches the
# documented version of PyTransit.
nbsphinx_execute = 'always'
nbsphinx_timeout = 900
nbsphinx_allow_errors = False

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = f'PyTransit {version}'

html_theme_options = {
    'repository_url': 'https://github.com/hpparvi/PyTransit',
    'use_repository_button': True,
    'use_issues_button': True,
    'use_download_button': False,
    'path_to_docs': 'docs/source',
    'home_page_in_toc': True,
}
