import os
import pkgutil
from jinja2 import Environment, FileSystemLoader

# Define paths
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "../templates")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../source/api/generated")
PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mufasa"))

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# Template file
TEMPLATE_NAME = "index.rst.template"
OUTPUT_NAME = "index.rst"

def discover_package(package_dir):
    """Discover subpackages and modules in a package directory."""
    subpackages = []
    modules = []

    for importer, name, is_pkg in pkgutil.walk_packages([package_dir], prefix="mufasa."):
        if is_pkg:
            subpackages.append(name)
        else:
            modules.append(name)

    return {"subpackages": sorted(subpackages), "modules": sorted(modules)}

def render_index(api_structure):
    """Render the index.rst using the Jinja2 template."""
    # Load the template
    template = env.get_template(TEMPLATE_NAME)
    # Render the output
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    with open(output_path, "w") as f:
        f.write(template.render(api_structure=api_structure))
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    api_structure = discover_package(PACKAGE_DIR)
    render_index(api_structure)
