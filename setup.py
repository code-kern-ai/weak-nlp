#!/usr/bin/env python
import os

from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as file:
    long_description = file.read()

setup(
    name="weak-nlp",
    version="0.0.2",
    author="Johannes Hötter",
    author_email="johannes.hoetter@kern.ai",
    description="Intelligent information integration based on weak supervision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/code-kern-ai/weak-nlp",
    keywords=["kern.ai", "machine learning", "supervised learning", "python"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"": "."},
    packages=find_packages("."),
    install_requires=[
        "numpy==1.21.4",
        "pandas==1.3.4"
    ],
)