import os
import pkgutil
from jinja2 import Environment, FileSystemLoader

# Paths
PACKAGE_DIR = os.path.abspath("../mufasa")  # Path to your Python package
TEMPLATE_DIR = os.path.abspath("templates")  # Path to the Jinja2 templates
OUTPUT_FILE = os.path.abspath("source/api/index.rst")  # Output file for index.rst
GENERATED_DIR = os.path.abspath("source/api/generated")  # Location of generated .rst files

# Jinja2 setup
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
template = env.get_template("index.rst.template")

def discover_modules(package_dir, package_name="mufasa"):
    """Discover all modules and submodules in the package."""
    modules = []
    for importer, modname, ispkg in pkgutil.walk_packages([package_dir], prefix=f"{package_name}."):
        modules.append(modname)
    return sorted(modules)

def get_module_description(module_name):
    """Generate a placeholder or fetch a real description for a module."""
    # Ideally, you extract the docstring of the module dynamically here.
    return f"Description for {module_name}"

def write_index_rst(modules):
    """Generate index.rst dynamically from the template."""
    module_data = [(module, get_module_description(module)) for module in modules]
    rendered = template.render(MODULES=module_data)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(rendered)
    print(f"Generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    modules = discover_modules(PACKAGE_DIR)
    write_index_rst(modules)
