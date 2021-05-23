import hashlib
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import Union
from urllib.request import urlopen
from urllib.parse import urlparse
from zipfile import ZipFile


PRETRAINED_MODEL_MAP = {
    'cnn': 'https://github.com/esceptico/toxic/releases/download/v0.1.3/model.pth.zip'
}
CACHE_DIR = Path.home() / '.toxic_models'
MODEL_FILE_NAME = 'model.pth'


def is_remote_url(url_or_filename: str) -> bool:
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def download(url: str, save_dir: Union[str, Path]):
    with urlopen(url) as response, NamedTemporaryFile() as tmp_file:
        copyfileobj(response, tmp_file)
        with ZipFile(tmp_file) as archive:
            for name in archive.namelist():
                target = Path(save_dir) / name
                with archive.open(name) as source:
                    with open(target, 'wb') as destination:
                        copyfileobj(source, destination)


def load_pretrained(model_name: str):
    try:
        url = PRETRAINED_MODEL_MAP[model_name]
    except KeyError:
        if is_remote_url(model_name):
            raise ValueError('Direct downloading is not supported yet.')
        raise ValueError(f'{model_name} model is not supported.')
    cache_name = hashlib.md5(model_name.encode('utf-8')).hexdigest()
    cache_path = CACHE_DIR / cache_name
    cached_model = cache_path / MODEL_FILE_NAME
    if cached_model.exists():
        return cached_model
    cache_path.mkdir(parents=True, exist_ok=True)
    download(url, cache_path)
    return cached_model
