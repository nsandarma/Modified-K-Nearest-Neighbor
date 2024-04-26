from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

DESCRIPTION = "Modified K-Nearest Neighbor"

NAME = "mknn"
VERSION = "0.0.2"
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nsandarma/Modified-K-Nearest-Neighbor",
    author="nsandarma",
    author_email="nsandarma@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=['scikit-learn','numpy'],
)
