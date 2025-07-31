from setuptools import setup, find_packages

setup(
    name="gluecore",
    version="3.1.0",
    packages=find_packages(),
    description="Minimalistic component composition interface",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/gluecore",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.7",
    keywords="composition, components, glue, minimal, architecture",
    project_urls={
        "Documentation": "https://github.com/yourusername/gluecore/wiki",
        "Source": "https://github.com/yourusername/gluecore",
        "Tracker": "https://github.com/yourusername/gluecore/issues",
    },
)