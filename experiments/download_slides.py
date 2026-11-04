"""Optional course WSI assets for fresh attention inference, about 4.95 GB.

Default notebook replay displays the saved figures instead. This downloader
resumes partial files and never replaces an existing file with different data.


"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(path, item):
    """Check size and content hashes; return False for an incomplete file."""
    if path.stat().st_size != item['size']:
  
        return False
  
    sha = hashlib.sha256()
    blob = hashlib.sha1(b'blob ' + str(item['size']).encode() + b'\0')
  
    with path.open('rb') as stream:
  
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            sha.update(block)
            blob.update(block)
  
  
    return ((not item.get('sha256') or sha.hexdigest() == item['sha256'])
            and (not item.get('git_blob_sha1') or blob.hexdigest() == item['git_blob_sha1']))


def main():
  
    manifest = json.loads((ROOT / 'experiments/slide_manifest.json').read_text())
  
    if shutil.disk_usage(ROOT).free < sum(x['size'] for x in manifest) + 10 * 1024**3:
        raise SystemExit('Need download size plus 10 GiB free')
  
    for item in manifest:
  
        dest = ROOT / 'Project_A_Starter_Code' / item['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
  
        if dest.exists():
  
            if not verify(dest, item):
  
                raise SystemExit(f'Existing asset differs; refusing overwrite: {dest}')
  
        else:
  
            part = Path(str(dest) + '.part')
            subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                            '--continue-at', '-', '--output', str(part), item['url']], check=True)
  
            if not verify(part, item):
                raise SystemExit(f'Integrity mismatch: {part}')
            part.rename(dest)
  
  
        print('Verified', dest)


if __name__ == '__main__':
    main()
