from setuptools import setup, find_packages

setup(
    name="synthetic-timeseries-generation",
    version="0.1.0",
    description="Gaussian Process-based synthetic time series generation for stock price forecasting",
    author="Markus Research Ops",
    author_email="markus.research.ops@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pyarrow>=6.0.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "test": ["pytest>=6.0", "pytest-cov"],
        "torch": ["torch>=1.10.0"],  # optional for faster GP inference
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "generate-gp-synthetic=synthetic_timeseries_generation.cli:main",
        ],
    },
)
