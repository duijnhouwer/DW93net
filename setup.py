# -*- coding: utf-8 -*-
"""
Created on Thu May 22 21:53:35 2025

@author: jduij
"""

from pathlib import Path
from setuptools import setup, find_packages

setup(
    name='dw93net',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=Path(__file__).with_name('requirements.txt').read_text().splitlines(),
)
