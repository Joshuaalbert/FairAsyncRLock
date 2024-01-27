#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='fair_async_rlock',
      version='1.0.7',
      description='A well-tested implementation of a fair asynchronous RLock for concurrent programming.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/Joshuaalbert/FairAsyncRLock",
      author='Joshua G. Albert',
      author_email='albert@strw.leidenuniv.nl',
      setup_requires=[],
      install_requires=[],
      tests_require=[
          'pytest',
          'pytest-asyncio'
      ],
      package_dir={'': './'},
      packages=find_packages('./'),
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.7',
      )
