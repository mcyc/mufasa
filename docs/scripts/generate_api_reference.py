import os
import pkgutil
import inspect
import importlib

ROOT_PACKAGE = "mufasa"
OUTPUT_FILE = "scripts/api_reference.py"


def discover_package(package_name):
    """Discover all modules, classes, and functions in a package."""
    print(f"package name: {package_name}")
    package = importlib.import_module(package_name)
    package_dir = os.path.dirname(package.__file__)
    api_reference = {}

    for _, module_name, is_pkg in pkgutil.walk_packages([package_dir], prefix=f"{package_name}."):
        # Exclude private modules starting with "_"
        if module_name.split('.')[-1].startswith('_'):
            print(f"Skipping private module: {module_name}")
            continue

        print(f"module: {module_name}")

        try:
            module = importlib.import_module(module_name)

            # Discover classes and functions defined in the module
            classes = [
                cls for cls, obj in inspect.getmembers(module, inspect.isclass)
                if obj.__module__ == module_name and not cls.startswith("_")
            ]
            functions = [
                func for func, obj in inspect.getmembers(module, inspect.isfunction)
                if obj.__module__ == module_name and not func.startswith("_")
            ]

            # Add the module itself as a "module" type member if it has no classes or functions
            if not classes and not functions:
                members = [{"name": module_name.split('.')[-1], "type": "module"}]
            else:
                members = [
                    {"name": cls, "type": "class"} for cls in classes
                ] + [
                    {"name": func, "type": "function"} for func in functions
                ]

            api_reference[module_name] = {
                "module": module_name,
                "description": "No description available.",  # Placeholder description
                "members": members,
            }
        except ImportError as e:
            print(f"Error: Could not import {module_name} - {e}")
            continue

    return api_reference



def build_api_reference(package_name, package_dir=None):
    """Recursively build the API_REFERENCE dictionary."""
    package = importlib.import_module(package_name)
    if package_dir is None:
        package_dir = os.path.dirname(package.__file__)

    api_reference = {}

    # Discover members of the current package
    for _, module_name, is_pkg in pkgutil.iter_modules([package_dir], prefix=f"{package_name}."):
        # Skip private modules
        if module_name.split('.')[-1].startswith('_'):
            print(f"Skipping private module: {module_name}")
            continue

        try:
            module = importlib.import_module(module_name)

            # Discover classes and functions defined in the module
            classes = [
                cls for cls, obj in inspect.getmembers(module, inspect.isclass)
                if obj.__module__ == module_name and not cls.startswith("_")
            ]
            functions = [
                func for func, obj in inspect.getmembers(module, inspect.isfunction)
                if obj.__module__ == module_name and not func.startswith("_")
            ]

            # Add current module to the API reference
            api_reference[module_name] = {
                "module": module_name,
                "description": "No description available.",  # Placeholder description
                "members": [
                    {"name": cls, "type": "class"} for cls in classes
                ] + [
                    {"name": func, "type": "function"} for func in functions
                ],
            }

            # If the module is a package, recursively discover its submodules
            if is_pkg:
                sub_package_dir = os.path.join(package_dir, module_name.split('.')[-1])
                sub_api_reference = build_api_reference(module_name, sub_package_dir)
                api_reference.update(sub_api_reference)
                # Add submodules as "module" type members
                api_reference[module_name]["members"].extend(
                    {"name": submodule, "type": "module"} for submodule in sub_api_reference.keys()
                )
        except ImportError as e:
            print(f"Error: Could not import {module_name} - {e}")
            continue

    return api_reference



def write_api_reference(api_reference, output_file):
    """Write the API_REFERENCE dictionary to a Python file."""
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


if __name__ == "__main__":
    api_reference = build_api_reference(ROOT_PACKAGE)
    write_api_reference(api_reference, OUTPUT_FILE)
    print(f"Generated: {OUTPUT_FILE}")
