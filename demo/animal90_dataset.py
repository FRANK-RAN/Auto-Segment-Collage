import gdown
import zipfile
import os

print("Current working directory:", os.getcwd())
url = 'https://drive.google.com/uc?id=1XykJthWV9bWpzwccaBOBtc0aNJ6YI-94'
output_path = '/animal90.zip'

destination_directory = 'animals/'

os.makedirs(destination_directory, exist_ok=True)

gdown.download(url, output_path, quiet=False)

# Unzip the files
zip_file_path = 'data/animal90.zip'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for file in zip_ref.namelist():
        if file.startswith('animals/animals/'):
            new_path = os.path.join(destination_directory, file.replace('animals/animals/', 'animals/'))
            zip_ref.extract(file, new_path)
