from setuptools import setup
import tomli
import os
import re

# Paths
METADATA_FILE = os.path.join(os.path.dirname(__file__), "mufasa/_metadata.py")
PYPROJECT_FILE = os.path.join(os.path.dirname(__file__), "pyproject.toml")

def get_metadata():
    """Read metadata from _metadata.py."""
    metadata = {}
    with open(METADATA_FILE, "r") as f:
        for line in f:
            match = re.match(r"^__(\w+)__\s*=\s*[\"'](.+?)[\"']", line)
            if match:
                key, value = match.groups()
                metadata[key] = value
    return metadata

def get_dependencies():
    """Load dependencies and optional dependencies from pyproject.toml."""
    with open(PYPROJECT_FILE, "rb") as f:
        pyproject_data = tomli.load(f)
    project = pyproject_data.get("project", {})
    dependencies = project.get("dependencies", [])
    optional_dependencies = project.get("optional-dependencies", {})
    return dependencies, optional_dependencies

# Load metadata and dependencies
metadata = get_metadata()
dependencies, optional_dependencies = get_dependencies()

if __name__ == "__main__":
    setup(
        name=metadata.get("project"),
        version=metadata.get("version"),
        description="MUlti-component Fitter for Astrophysical Spectral Applications",
        author=metadata.get("author"),
        url=metadata.get("github_url"),
        packages=["mufasa"],
        install_requires=dependencies,
        extras_require=optional_dependencies,
        python_requires=">=3.8",
    )
