"""Install individual, checksum-verified checkpoint assets (no ZIP files)."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from nepal_cnn.utils.paths import PROJECT_ROOT as ROOT


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verified(path, item):
    return path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="ghimiredhikura/GeoBASP")
    parser.add_argument("--tag", default="v1.0.0", help="Published GitHub release tag (default: v1.0.0)")
    parser.add_argument("--assets-dir", type=Path, help="Install from local individual .pt files")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "metadata/models.json").read_text(encoding="utf-8"))
    if args.list:
        for item in catalog["models"]:
            print(f'{item["asset"]} -> {item["path"]} ({item["bytes"]/1024**2:.1f} MiB)')
        return
    if len(args.repo.split("/")) != 2 or any(not p or p in (".", "..") for p in args.repo.split("/")):
        parser.error("--repo must be OWNER/NAME")
    for item in catalog["models"]:
        target = (ROOT / item["path"]).resolve()
        if not target.is_relative_to((ROOT / "models").resolve()):
            raise ValueError("Unsafe model path")
        if target.exists():
            if not verified(target, item):
                raise ValueError(f"Existing model differs; preserve it before installing: {target}")
            print(f'Verified {item["path"]}')
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".pt.part")
        if args.assets_dir:
            source = args.assets_dir.resolve() / item["asset"]
            if not verified(source, item):
                raise ValueError(f"Asset missing or checksum differs: {source}")
            shutil.copyfile(source, temporary)
        else:
            owner, name = args.repo.split("/")
            url = (f"https://github.com/{quote(owner)}/{quote(name)}/releases/download/"
                   f'{quote(args.tag, safe="")}/{quote(item["asset"], safe="")}')
            request = Request(url, headers={"User-Agent": "GeoBASP-release"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, 8 * 1024 * 1024)
        if not verified(temporary, item):
            raise ValueError(f'Download/copy checksum differs: {item["asset"]}')
        temporary.replace(target)
        print(f'Installed {item["path"]}', flush=True)


if __name__ == "__main__":
    main()
