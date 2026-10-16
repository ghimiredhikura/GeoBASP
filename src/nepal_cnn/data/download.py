"""Download the original inventory and imagery/DEM inputs, verifying SHA-256 hashes."""

from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT
import argparse
import json
import hashlib
from pathlib import Path
import shutil
from urllib.request import Request, urlopen
from urllib.parse import quote


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['inventory','sentinel2','dem','all'], default='all')
    parser.add_argument('--tag', help='Inventory release tag; defaults to the data catalog')
    parser.add_argument('--assets-dir', type=Path, help='Copy inventory files from a local release-assets folder')
    parser.add_argument('--all-bands', action='store_true', help='Also download unused display/reference bands')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    groups = ['inventory','sentinel2','dem'] if args.group == 'all' else [args.group]
    tasks = []
    sources = json.loads((ROOT / "metadata/data_sources.json").read_text(encoding="utf-8"))
    for group in groups:
        if group == 'inventory':
            inventory = sources['inventory']
            tag = args.tag or inventory['release_tag']
            prefix = f'https://github.com/{inventory["repository"]}/releases/download/{quote(tag, safe="")}/'
            tasks.extend({**row, 'url': prefix + quote(row['asset'], safe='')}
                         for row in inventory['release_files'])
            continue
        rows = sources[group]
        if group == 'sentinel2' and not args.all_bands:
            rows = [row for row in rows if row['filename'] in ['B04.tif','B08.tif','B11.tif','SCL.tif','granule_metadata.xml','tile_metadata.xml']]
        tasks.extend(rows)
    if args.list:
        print(f'{len(tasks)} source assets, {sum(int(x["bytes"]) for x in tasks)/1024**3:.2f} GiB; no downloads performed.')
        return
    for index, task in enumerate(tasks, 1):
        target = (ROOT/task['path']).resolve()
        if not target.is_relative_to(ROOT/'data/raw'):
            raise ValueError(f'Unsafe data path: {task["path"]}')
        expected = task['sha256']
        if target.exists():
            if target.stat().st_size != int(task['bytes']) or digest(target) != expected:
                raise ValueError(f'Existing source differs: {target}')
        else:
            if args.verify_only:
                raise FileNotFoundError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.part')
            if args.assets_dir and task.get('asset'):
                source = args.assets_dir.resolve() / task['asset']
                if source.stat().st_size != int(task['bytes']) or digest(source) != expected:
                    raise ValueError(f'Local inventory asset differs: {source}')
                shutil.copyfile(source, temporary)
            else:
                request = Request(task['url'], headers={'User-Agent':'GeoBASP-release'})
                with urlopen(request,timeout=60) as response, temporary.open('wb') as output:
                    shutil.copyfileobj(response,output,8*1024*1024)
            if temporary.stat().st_size != int(task['bytes']) or digest(temporary) != expected:
                raise ValueError(f'Download verification failed: {target}')
            temporary.replace(target)
        print(f'[{index}/{len(tasks)}] Verified {task["path"]}',flush=True)


if __name__ == '__main__':
    main()
