import os
from jinja2 import Environment, FileSystemLoader
from api_reference import API_REFERENCE

# Paths and template setup
TEMPLATE_DIR = "templates"
MODULE_TEMPLATE = "module.rst.template"
CLASS_TEMPLATE = "class.rst.template"
FUNCTION_TEMPLATE = "function.rst.template"
INDEX_TEMPLATE = "index.rst.template"  # Template for api/index.rst
OUTPUT_DIR = "source/api" #"source/api/generated"
INDEX_OUTPUT = "source/api/index.rst"

env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))


def render_template(template_name, context):
    """Render a Jinja2 template with the given context."""
    template = env.get_template(template_name)
    return template.render(context)


def write_file(output_path, content):
    """Write content to the specified file path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)


def transform_api_reference(api_reference):
    """Transform API_REFERENCE to Scikit-learn-style format with type mapping."""
    transformed_reference = {}
    for module_name, module_details in api_reference.items():
        # Extract autosummary and type mapping
        members = module_details.get("members", [])
        autosummary = [member["name"] for member in members]
        types = {member["name"]: member["type"] for member in members}

        # Build transformed entry
        transformed_reference[module_name] = {
            "short_summary": module_details.get("description", "No description available."),
            "description": None,  # Add detailed descriptions if available
            "sections": [
                {
                    "title": None,  # Titles can be added if needed
                    "autosummary": autosummary,
                }
            ],
            "type": types,  # Include type mapping
        }
    return transformed_reference


def generate_api_index():
    """Generate the api/index.rst file."""
    transformed_reference = transform_api_reference(API_REFERENCE)
    context = {
        "API_REFERENCE": transformed_reference,
        "DEPRECATED_API_REFERENCE": [],  # Add deprecated items if applicable
    }
    content = render_template(INDEX_TEMPLATE, context)
    write_file(INDEX_OUTPUT, content)
    print(f"Generated index .rst: {INDEX_OUTPUT}")


def generate_module_rst(module_name, module_details):
    """Generate .rst file for a module."""
    members = module_details["members"]

    context = {
        "full_name": module_details["module"],
        "short_name": module_details["module"].split('.')[-1],
        "docstring": module_details.get("description", "No description available."),
        "submodules": [m["name"] for m in members if m["type"] == "module"],
        "classes": [f"{module_details['module']}.{m['name']}" for m in members if m["type"] == "class"],
        "functions": [f"{module_details['module']}.{m['name']}" for m in members if m["type"] == "function"],
    }

    output_path = os.path.join(OUTPUT_DIR, f"{module_name}.rst")
    content = render_template(MODULE_TEMPLATE, context)
    write_file(output_path, content)
    print(f"Generated module .rst: {output_path}")


def generate_member_rst(module_name, member):
    """Generate .rst file for a class, function, or submodule."""
    if member["type"] == "class":
        template_name = CLASS_TEMPLATE
    elif member["type"] == "function":
        template_name = FUNCTION_TEMPLATE
    else:
        raise ValueError(f"Unknown member type: {member['type']}")

    # Prepare the context for rendering
    context = {
        "full_name": f"{module_name}.{member['name']}",
        "short_name": member["name"],
        "docstring": f"Documentation for {member['name']}.",  # Replace with actual docstring if available
    }

    # Output .rst file in the `generated/` directory
    output_path = os.path.join(OUTPUT_DIR, f"{module_name}.{member['name']}.rst")
    content = render_template(template_name, context)
    write_file(output_path, content)
    print(f"Generated member .rst: {output_path}")


def generate_rst_files(generate_member = False):
    """Generate all .rst files for modules, classes, and functions."""

    # Step 1: Generate module and member .rst files
    for module_name, module_details in API_REFERENCE.items():
        generate_module_rst(module_name, module_details)
        if generate_member:
            for member in module_details["members"]:
                if member["type"] != "module":  # Skip submodules for member-level files
                    generate_member_rst(module_name, member)

    # Step 2: Generate the index.rst
    generate_api_index()


if __name__ == "__main__":
    generate_rst_files()
