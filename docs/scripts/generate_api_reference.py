import os
import pkgutil
import inspect
import importlib

ROOT_PACKAGE = "mufasa"
OUTPUT_FILE = "scripts/api_reference.py"

def discover_package(package_name):
    """Discover all modules, classes, functions, and submodules in a package."""
    package = importlib.import_module(package_name)
    package_dir = os.path.dirname(package.__file__)
    api_reference = {}

    # Walk through the package directory to find modules and submodules
    for _, module_name, is_pkg in pkgutil.walk_packages([package_dir], prefix=f"{package_name}."):
        # Skip modules starting with "._"
        if '._' in module_name:
            print(f"Skipping hidden module: {module_name}")
            continue

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            print(f"Error: Could not import {module_name} - {e}")
            continue

        # Discover classes and functions defined in the module
        classes = [
            cls for cls, obj in inspect.getmembers(module, inspect.isclass)
            if obj.__module__ == module_name
        ]
        functions = [
            func for func, obj in inspect.getmembers(module, inspect.isfunction)
            if obj.__module__ == module_name
        ]

        # Add submodules explicitly for packages
        if is_pkg:
            submodules = [
                submodule_name for _, submodule_name, sub_is_pkg in pkgutil.iter_modules(
                    [os.path.dirname(module.__file__)], prefix=f"{module_name}."
                )
                if '._' not in submodule_name  # Skip hidden submodules
            ]
            members = [{"name": submodule, "type": "module"} for submodule in submodules]
        else:
            members = []

        # Add classes and functions to members
        members += [{"name": cls, "type": "class"} for cls in classes]
        members += [{"name": func, "type": "function"} for func in functions]

        # Store module details in the API reference
        api_reference[module_name] = {
            "module": module_name,
            "description": "No description available.",  # Placeholder description
            "members": members,
        }

    return api_reference


def write_api_reference(api_reference, output_file):
    """Write the API_REFERENCE dictionary to a Python file with confirmation."""
    if os.path.exists(output_file):
        # Prompt user before overwriting
        confirm = input(f"{output_file} exists. Overwrite? (y/n): ")
        if confirm.lower() != 'y':
            print("Aborting...")
            return

    with open(output_file, "w") as f:
        f.write("# This file is auto-generated. Edit descriptions and structure as needed.\n\n")
        f.write("API_REFERENCE = {\n")
        for module_name, details in api_reference.items():
            f.write(f"    '{module_name}': {{\n")
            f.write(f"        'module': '{details['module']}',\n")
            f.write(f"        'description': '{details['description']}',\n")
            f.write(f"        'members': [\n")
            for member in details["members"]:
                f.write(f"            {{'name': '{member['name']}', 'type': '{member['type']}'}}")
                if member != details["members"][-1]:  # Avoid trailing commas
                    f.write(",\n")
            f.write("\n        ]\n")
            f.write("    },\n")
        f.write("}\n")

    print(f"Generated: {output_file}")


if __name__ == "__main__":
    api_reference = discover_package(ROOT_PACKAGE)
    write_api_reference(api_reference, OUTPUT_FILE)
