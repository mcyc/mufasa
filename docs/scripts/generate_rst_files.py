import os
from jinja2 import Environment, FileSystemLoader
from api_reference import API_REFERENCE

# Paths and template setup
TEMPLATE_DIR = "templates"
MODULE_TEMPLATE = "module.rst.template"
CLASS_TEMPLATE = "class.rst.template"
FUNCTION_TEMPLATE = "function.rst.template"
OUTPUT_DIR = "source/api/generated"

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


def generate_module_rst(module_name, module_details):
    """Generate .rst file for a module."""
    # Separate out classes, functions, and submodules
    classes = [
        member["name"] for member in module_details["members"] if member["type"] == "class"
    ]
    functions = [
        member["name"] for member in module_details["members"] if member["type"] == "function"
    ]
    submodules = [
        member["name"] for member in module_details["members"] if member["type"] == "module"
    ]

    # Prepare the context for rendering
    context = {
        "full_name": module_details["module"],  # Full module name
        "short_name": module_details["module"].split('.')[-1],  # Short name (last part)
        "docstring": module_details.get("description", "No description available."),
        "classes": classes,
        "functions": functions,
        "submodules": submodules,  # Pass full names; template handles short names
    }

    # Render and write the .rst file
    output_path = os.path.join(OUTPUT_DIR, f"{module_name}.rst")
    content = render_template(MODULE_TEMPLATE, context)
    write_file(output_path, content)
    print(f"Generated module .rst: {output_path}")



def generate_member_rst(module_name, member):
    """Generate .rst file for a class or function."""
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




def generate_rst_files():
    """Generate .rst files for all modules and members defined in API_REFERENCE."""
    for module_name, module_details in API_REFERENCE.items():
        generate_module_rst(module_details["module"], module_details)
        for member in module_details["members"]:
            if member["type"] != "module":  # Skip submodules for member-level files
                generate_member_rst(module_details["module"], member)


if __name__ == "__main__":
    generate_rst_files()
