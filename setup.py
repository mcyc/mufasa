import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mufasa", # Replace with your own username
    version="0.1.1",
    author="Michael Chun-Yuan Chen",
    author_email="mcychen@uvic.ca",
    description="MUlti-component Fitter for Astrophysical Spectral Applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcyc/mufasa",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)