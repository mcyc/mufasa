import os
from jinja2 import Environment, FileSystemLoader
from generate_api_reference import build_api_reference

# Define paths
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "../templates")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../source/api")
PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mufasa"))

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# Template files
INDEX_TEMPLATE_NAME = "index.rst.template"
MODULES_TEMPLATE_NAME = "modules.rst.template"

def render_index(api_reference):
    """Render the index.rst using the Jinja2 template."""
    template = env.get_template(INDEX_TEMPLATE_NAME)
    output_path = os.path.join(OUTPUT_DIR, "index.rst")
    with open(output_path, "w") as f:
        f.write(template.render(API_REFERENCE=api_reference))
    print(f"Generated: {output_path}")

def render_modules(api_reference):
    """Render the modules.rst using the Jinja2 template."""
    template = env.get_template(MODULES_TEMPLATE_NAME)
    output_path = os.path.join(OUTPUT_DIR, "modules.rst")
    with open(output_path, "w") as f:
        f.write(template.render(API_REFERENCE=api_reference))
    print(f"Generated: {output_path}")

if __name__ == "__main__":
    # Generate the API_REFERENCE dictionary
    api_reference = build_api_reference(PACKAGE_DIR, "mufasa")

    # Render the index.rst file
    render_index(api_reference)

    # Render the modules.rst file
    render_modules(api_reference)
