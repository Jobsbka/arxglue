from setuptools import setup, find_packages

setup(
    name="arxglue",
    version="1.0.1",
    packages=find_packages(),
    description="Minimalistic component composition interface",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="VKB Arcghitector",
    url="https://github.com/jobsbka/gluecore",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",

        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
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
        "Source": "https://github.com/jobsbka/gluecore",
    },
)