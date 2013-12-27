#!/usr/bin/env python

import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

print(find_packages())
setup(
    name = 'ipcv',
    version = '0.1',
    author = 'Anders Boesen Lindbo Larsen',
    author_email = 'abll@dtu.dk',
    description = "ABLL's personal image processing/computer vision toolkit.",
    license = 'MIT',
    url = 'http://compute.dtu.dk/~abll',
    packages = find_packages(),
    install_requires = ['numpy', 'scipy', 'matplotlib'],
    long_description = read('README.md'),
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
)

