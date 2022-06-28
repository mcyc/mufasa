import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mufasa", 
    version="1.0.1",
    author="Michael Chun-Yuan Chen",
    author_email="chen.m@queensu.ca",
    description="MUlti-component Fitter for Astrophysical Spectral Applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcyc/mufasa",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
