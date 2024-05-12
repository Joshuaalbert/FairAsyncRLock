#!/usr/bin/env python

import os

from setuptools import setup

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

tests_requirement_path = f"{lib_folder}/requirements-tests.txt"
tests_require = []
if os.path.isfile(tests_requirement_path):
    with open(tests_requirement_path) as f:
        tests_require = f.read().splitlines()

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="fair_async_rlock",
    version="1.1.0",
    description="A well-tested implementation of a fair asynchronous RLock for concurrent programming.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Joshuaalbert/FairAsyncRLock",
    author="Joshua G. Albert",
    author_email="albert@strw.leidenuniv.nl",
    setup_requires=[],
    install_requires=install_requires,
    tests_require=tests_require,
    package_dir={"fair_async_rlock": "./fair_async_rlock"},
    packages=["fair_async_rlock"],
    package_data={"fair_async_rlock": ["py.typed"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Framework :: AnyIO",
        "Typing :: Typed",
    ],
    python_requires=">=3.11",
)
