import os
import re

'''
Renames the api/*.rst files to remove "mufasa." prefix in filenames but ensures module references in the content are correct.
'''

api_dir = "/Users/mcychen/Documents/GitRepos/My_Public_Repos/mufasa/docs/source/api"

for filename in os.listdir(api_dir):
    if filename.startswith("mufasa."):
        # Rename file to remove 'mufasa.' prefix
        new_name = filename.replace("mufasa.", "", 1)
        os.rename(os.path.join(api_dir, filename), os.path.join(api_dir, new_name))

        # Edit the content of the file
        file_path = os.path.join(api_dir, new_name)
        with open(file_path, 'r') as file:
            content = file.read()

        # Only remove 'mufasa.' prefix in filenames, but keep it in `.. automodule::` directives
        content = re.sub(r'^(\s*\.\.\s*automodule::\s*)mufasa\.', r'\1mufasa.', content, flags=re.MULTILINE)

        # Write the updated content back to the file
        with open(file_path, 'w') as file:
            file.write(content)


