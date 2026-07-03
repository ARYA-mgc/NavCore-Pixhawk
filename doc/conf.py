import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'NavCore-Pixhawk'
copyright = '2026, NavCore-Pixhawk Contributors'
author = 'NavCore-Pixhawk Contributors'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
