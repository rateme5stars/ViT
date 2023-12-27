from pathlib import Path
import urllib
import os 
import requests
import yaml
from tqdm import tqdm
import zipfile
import tarfile

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(os.path.dirname(current_dir))
config_path = os.path.join(parent_dir, 'config.yaml')

with open(config_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

def is_url(url, check=True):
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False

def download(url: str, unzip: bool):
    if not is_url(url):
        print('This is not a valid URL')
        return

    DATASET_PATH = Path('data') 
    DATASET_PATH.mkdir(parents=True, exist_ok=True)

    filename = url.split('/')[-1]
    file_path = DATASET_PATH / filename

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))

    print(f'Starting download of {filename}')
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            progress_bar.update(len(chunk))
            f.write(chunk)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    print(f'Download completed and saved to {file_path}')
    
    if unzip:
        unzip(parent_dir / file_path)

def unzip(path: Path):

    if not path.exists():
        print(f"The file {path} does not exist.")
        return

    print('Starting extracting dataset...')

    if path.suffix == '.zip':
        with zipfile.ZipFile(path, 'r') as zip_ref:
            extract_dir = parent_dir / 'data'
            zip_ref.extractall(extract_dir)
            print(f"Extracted to: {extract_dir}")

    elif path.name.endswith('.tar.gz'):
        with tarfile.open(path, 'r:gz') as tar:
            extract_dir = parent_dir / 'data'
            tar.extractall(extract_dir)
            print(f"Extracted to {extract_dir}")

if __name__ == '__main__':
    download(yaml_data['data']['url'], True)
    