import zipfile
from pathlib import Path

# Set the target directory
p = Path('.')

# Iterate over all Zip files in the directory
for f in p.glob('*.zip'):
    
    with zipfile.ZipFile(f, 'r') as archive:
        # Extract all contents of the Zip file to a directory with the same name as the file
        archive.extractall(path=f'./{f.stem}')
        # Print a message indicating that the extraction is complete
        print(f"Extracted contents from '{f.name}' to '{f.stem}' directory.")