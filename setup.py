#!/usr/bin/env python3
"""Setup script for the hash_verifier package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hash_verifier",
    version="1.0.0",
    author="Damon Cadden",
    author_email="damonc2013@gmail.com",
    description="Utility for verifying file integrity using SHA-256 hashes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/damonc0313/FlameMirror-Canonical2.0",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hash-verifier=hash_verifier:main",
        ],
    },
)