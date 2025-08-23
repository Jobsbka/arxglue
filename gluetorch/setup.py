"""
Setup script for TransGlue libraries
"""

from setuptools import setup, find_packages

# Setup for TransGlueCore
setup(
    name="gluetorch",
    version="0.1.0",
    description="WIP",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/transglue",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

