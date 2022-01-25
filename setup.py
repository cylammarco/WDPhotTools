#!/usr/bin/env python3

import codecs
import os
import re
from setuptools import setup, find_packages

__packagename__ = "WDPhotTools"

META_PATH = os.path.join(__packagename__, "__init__.py")

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file,
        re.M)
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


install_requires = [
    'numpy>=1.21', 'scipy>=1.7', 'matplotlib>=3.5', 'emcee>=3.0', 'corner'
]

setup(name=__packagename__,
      version=find_meta("version"),
      packages=find_packages(),
      author=find_meta("author"),
      author_email=find_meta("email"),
      maintainer=find_meta("maintainer"),
      maintainer_email=find_meta("email"),
      url="https://github.com/cylammarco/WDPhotTools",
      license=find_meta("license"),
      description=find_meta("description"),
      long_description=read(os.path.join(HERE, "README.md")),
      long_description_content_type='text/markdown',
      zip_safe=False,
      include_package_data=True,
      install_requires=install_requires,
      python_requires='>=3.6')
