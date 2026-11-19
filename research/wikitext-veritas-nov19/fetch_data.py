"""Retrieve the frozen upstream training and validation files."""
import hashlib
import urllib.request
from pathlib import Path

REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
FILES = {
    'train': 'e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7',
    'validation': '204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c',
}
for split, expected in FILES.items():
    path = Path(__file__).resolve().parent / (split + '.parquet')
    url = f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Hash mismatch for {path}: {actual}')
    print(f'{path.name}: {actual}')
