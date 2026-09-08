# -*- coding: utf-8 -*-
"""
Created on Thu May 22 21:53:35 2025

@author: jduij
"""

from setuptools import setup, find_packages

setup(
    name='dw93net',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'torch'
    ],
)
