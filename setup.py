#!/usr/bin/env python3
"""
Setup script for Massive Scale Autonomous Evolution System
Billion-Parameter, Million-Line Autonomous Code Evolution
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    # Fallback requirements if file doesn't exist
    requirements = [
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'pandas>=1.3.0',
        'networkx>=2.6.0',
        'psutil>=5.8.0'
    ]

setup(
    name="massive-scale-evolution",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="Billion-Parameter, Million-Line Autonomous Code Evolution System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/massive-scale-evolution",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "visualization": [
            "plotly>=5.0",
            "seaborn>=0.11",
            "bokeh>=2.4",
        ],
        "distributed": [
            "dask>=2021.6.0",
            "ray>=1.4.0",
            "mpi4py>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "massive-evolution=autonomous_evolution_engine:main",
            "mega-generator=massive_scale_generator:main",
            "autonomous-demo=autonomous_demo:main",
            "theoretical-foundations=theoretical_foundations:main",
            "advanced-algorithms=advanced_evolutionary_algorithms:main",
        ],
    },
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "autonomous-ai",
        "billion-parameters",
        "evolutionary-algorithms",
        "distributed-computing",
        "code-evolution",
        "neural-networks",
        "optimization",
        "machine-learning",
        "artificial-intelligence",
        "massive-scale",
        "quality-diversity",
        "pareto-optimization",
        "autonomous-systems",
        "phd-research",
        "scientific-computing"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/massive-scale-evolution/issues",
        "Source": "https://github.com/yourusername/massive-scale-evolution",
        "Documentation": "https://github.com/yourusername/massive-scale-evolution/blob/main/README.md",
        "Research Paper": "https://github.com/yourusername/massive-scale-evolution/blob/main/docs/",
    },
)