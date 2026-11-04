"""Explicitly invoked only after large-download approval; resumable .part files, SHA256 check."""
import json,subprocess,hashlib,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

manifest=json.loads((ROOT/'experiments/feature_manifest.json').read_text())

if shutil.disk_usage(ROOT).free < max(40*10**9, sum(x['size'] for x in manifest)+10*1024**3): raise SystemExit('Insufficient free disk')

for x in sorted(manifest,key=lambda x: (0 if '/C16/' in x['path'] else 1 if '/C17/' in x['path'] else 2)):
    dest=ROOT/'Project_A_Starter_Code'/x['path']; dest.parent.mkdir(parents=True,exist_ok=True)

    def digest(path):

        sha=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''): sha.update(b)
        return sha.hexdigest()

    if dest.exists():
        if dest.stat().st_size==x['size'] and digest(dest)==x['sha256']: continue
        raise SystemExit(f'Existing file differs; refusing overwrite: {dest}')

    part=Path(str(dest)+'.part')

    subprocess.run(['curl','--fail','--location','--retry','3','--connect-timeout','30','--speed-limit','1024','--speed-time','120','--continue-at','-','--output',str(part),x['url']],check=True)

    if part.stat().st_size!=x['size'] or digest(part)!=x['sha256']: raise SystemExit('Size/checksum mismatch: '+str(part))

    part.rename(dest)


    print('Verified '+str(dest),flush=True)
