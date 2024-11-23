import os
import re

# Paths
GENERATED_DIR = "source/api/generated"
MUFASA_RST = os.path.join(GENERATED_DIR, "mufasa.rst")
OUTPUT_FILE = "source/api/index.rst"

def extract_toc_sections(filepath):
    """Extract Subpackages and Submodules sections with their entries from a .rst file."""
    with open(filepath, "r") as f:
        content = f.read()

    # Regex to match `Subpackages` and its entries, retaining indentation
    subpackages_match = re.search(
        r"Subpackages.*?\n.. toctree::.*?\n(?:\s+:maxdepth:.*?\n)?((?:\s{3,}\S.*?\n)+)",
        content,
        re.DOTALL,
    )
    submodules_match = re.search(
        r"Submodules.*?\n.. toctree::.*?\n(?:\s+:maxdepth:.*?\n)?((?:\s{3,}\S.*?\n)+)",
        content,
        re.DOTALL,
    )

    # Extract matches or default to empty strings
    subpackages_toc = subpackages_match.group(1).rstrip() if subpackages_match else ""
    submodules_toc = submodules_match.group(1).rstrip() if submodules_match else ""

    return subpackages_toc, submodules_toc


def parse_generated_modules(directory):
    """Parse generated .rst files and extract module names and descriptions."""
    modules = []
    for filename in os.listdir(directory):
        if filename.endswith(".rst") and (filename != "mufasa.rst" and filename != "modules.rst"):
            module_name = filename.replace(".rst", "")
            # Extract module description from the file or default to a placeholder
            with open(os.path.join(directory, filename)) as f:
                lines = f.readlines()
                description = next((line.strip() for line in lines if line.strip()), "No description available.")
            modules.append((module_name, description))
    return modules

def write_api_index(modules, subpackages_toc, submodules_toc):
    """Write api/index.rst with both a table and hierarchical sidebar entries."""
    def prepend_generated(toc):
        """Prepend 'generated/' to each line in the ToC."""
        return "\n".join(f"   generated/{line.strip()}" for line in toc.splitlines() if line.strip())

    with open(OUTPUT_FILE, "w") as f:
        # Title
        f.write("===============\nAPI Reference\n===============\n\n")
        f.write("This is the class and function reference of `MUFASA` API.\n\n")

        # Table of modules
        #f.write("Modules Table\n-------------\n\n")
        #f.write("This table lists all the modules and submodules in Mufasa, with links to their detailed documentation.\n\n")
        f.write(".. list-table::\n   :header-rows: 1\n\n")
        f.write("   * - Module\n     - Description\n")
        for module_name, description in modules:
            f.write(f"   * - :mod:`{module_name}`\n     - {description}\n")
        f.write("\n")

        # Write Subpackages ToC
        if subpackages_toc:
            #f.write("\nSubpackages\n-----------\n\n")
            f.write("\n")
            f.write(".. toctree::\n   :maxdepth: 4\n   :hidden:\n\n")
            f.write(prepend_generated(subpackages_toc) + "\n\n")

        # Write Submodules ToC
        if submodules_toc:
            #f.write("\nSubmodules\n----------\n\n")
            f.write("\n")
            f.write(".. toctree::\n   :maxdepth: 4\n   :hidden:\n\n")
            f.write(prepend_generated(submodules_toc) + "\n\n")

    print(f"Generated: {OUTPUT_FILE}")


if __name__ == "__main__":
    # Extract ToC entries for subpackages and submodules
    subpackages_toc, submodules_toc = extract_toc_sections(MUFASA_RST)

    # Generate the modules table
    modules = parse_generated_modules(GENERATED_DIR)

    # Write the combined index.rst
    write_api_index(modules, subpackages_toc, submodules_toc)
