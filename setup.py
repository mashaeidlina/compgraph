#!usr/bin/env python

from setuptools import setup

packages = {
    "graph": "graph/",
    "graph.src": "graph/src",
    "tests": "tests/"
}

setup(
    name="graph",
    version="0.1",
    description="Computational Graph",
    author="Masha Eidlina",
    author_email="mashaeidlina@yandex.ru",
    requires=['pytest'],
    # requires=['logging', 'json', 'itertools', 'typing', 'io'],
    packages=packages,
    packages_dir=packages
)
