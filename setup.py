"""
Setup script for ExoplanetHunter-TransitML package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="exoplanet-hunter-transitml",
    version="0.1.0",
    author="ExoplanetHunter Team",
    author_email="contact@exoplanethunter.com",
    description="Machine Learning for Exoplanet Detection from Transit Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/backlundtransform/ExoplanetHunter-TransitML",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/backlundtransform/ExoplanetHunter-TransitML/issues",
        "Source": "https://github.com/backlundtransform/ExoplanetHunter-TransitML",
    },
)