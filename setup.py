#!/usr/bin/env python3
"""
Setup script for assoG2P package
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# 读取requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "lightgbm>=3.3.0",
        "xgboost>=1.5.0",
        "catboost>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "kaleido>=0.2.1",
    ]

setup(
    name="assoG2P",
    version="1.0.0",
    author="chenrf",
    author_email="12024128035@stu.ynu.edu.cn",
    description="Genome-wide Association Analysis Toolkit with Machine Learning Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/assoG2P",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "full": [
            "shap>=0.40.0",
            "tqdm>=4.62.0",
            "psutil>=5.8.0",
            "datatable>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "association=assoG2P.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "assoG2P": [
            "bin/software/*",
        ],
    },
    zip_safe=False,
)
