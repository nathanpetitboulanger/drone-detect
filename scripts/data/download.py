"""Téléchargement des jeux de données bruts de détection de drones.

Sources (voir docs/DATA.md pour les licences) :
  - halmstad  : Svanström et al., « Drone-detection-dataset » (Zenodo 10.5281/zenodo.5500576,
                CC0-1.0). Archive 311 Mo ; on n'extrait que les vidéos visibles (Data/Video_V).
  - dut_track : DUT Anti-UAV, partie tracking (20 séquences, Google Drive, Apache-2.0), ~3,6 Go.
  - dut_det   : DUT Anti-UAV, partie détection (train/val/test officiels, Google Drive), ~1,4 Go.

Le script est idempotent : une archive déjà présente avec la bonne taille n'est pas
re-téléchargée, une extraction terminée est marquée par un fichier `.extracted`.

Usage :
    .venv/bin/python scripts/data/download.py                     # toutes les sources
    .venv/bin/python scripts/data/download.py --sources halmstad  # une seule source
    .venv/bin/python scripts/data/download.py --list              # tailles annoncées
    .venv/bin/python scripts/data/download.py --keep-archives     # garder les .zip
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# Chaque fichier : nom local, URL (http) ou identifiant Google Drive, taille attendue
# (octets, approximative pour Drive), md5 si connu, membres à extraire (préfixe) ou None.
SOURCES: dict[str, dict] = {
    "halmstad": {
        "license": "CC0-1.0",
        "files": [
            {
                "name": "Drone-detection-dataset-v1.0.0.zip",
                "url": "https://zenodo.org/api/records/5500576/files/"
                "DroneDetectionThesis/Drone-detection-dataset-v1.0.0.zip/content",
                "size": 311_348_350,
                "md5": "5d9b891a87857b3d9ee8872e8a1f0f0f",
                # On ne garde que le visible + la description des vidéos.
                "members": ["Data/Video_V/", "Data/Video_dataset_description.xlsx",
                            "LICENSE", "README.md"],
            }
        ],
    },
    "dut_track": {
        "license": "Apache-2.0 (licence du dépôt GitHub wangdongdut/DUT-Anti-UAV)",
        "files": [
            {"name": "Anti-UAV-Tracking-V0.zip", "gdrive": "1dlSPDggg6TRFMcC1jlYIJxxzUQS1mIh9",
             "size": 3_640_000_000, "members": None},
            {"name": "Anti-UAV-Tracking-V0GT.zip", "gdrive": "16PE3tBhT0lUGZLA8-zIRYvNUvxfhFZJq",
             "size": 90_836, "members": None},
        ],
    },
    "dut_det": {
        "license": "Apache-2.0 (licence du dépôt GitHub wangdongdut/DUT-Anti-UAV)",
        "files": [
            {"name": "train.zip", "gdrive": "1RVsSGPUKTdmoyoPTBTWwroyulLek1eTj",
             "size": 745_000_000, "members": None},
            {"name": "val.zip", "gdrive": "1333uEQfGuqTKslRkkeLSCxylh6AQ0X6n",
             "size": 372_000_000, "members": None},
            {"name": "test.zip", "gdrive": "1L1zeW1EMDLlXHClSDcCjl3rs_A6sVai0",
             "size": 271_000_000, "members": None},
        ],
    },
}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def archive_ok(path: Path, spec: dict) -> bool:
    """Vérifie qu'une archive locale est complète (md5 si connu, sinon zip lisible)."""
    if not path.exists():
        return False
    if spec.get("md5"):
        return md5sum(path) == spec["md5"]
    try:
        with zipfile.ZipFile(path) as z:
            return len(z.namelist()) > 0
    except zipfile.BadZipFile:
        return False


def fetch(spec: dict, dest: Path) -> None:
    """Télécharge un fichier (HTTP direct ou Google Drive via gdown) vers `dest`."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    if "gdrive" in spec:
        import gdown  # import local : dépendance seulement utile pour Drive

        gdown.download(id=spec["gdrive"], output=str(tmp), quiet=False, resume=True)
    else:
        print(f"  GET {spec['url']}")
        with urllib.request.urlopen(spec["url"]) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)


def extract(archive: Path, out_dir: Path, members: list[str] | None) -> None:
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if members is not None:
            # Les archives GitHub/Zenodo ont un dossier racine « <repo>-<tag>/ » : on le retire.
            root = names[0].split("/")[0] + "/"
            keep = [n for n in names if any(n[len(root):].startswith(m) for m in members)]
            for n in keep:
                rel = n[len(root):]
                target = out_dir / rel
                if n.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            print(f"  extrait {len(keep)} entrées / {len(names)}")
        else:
            z.extractall(out_dir)
            print(f"  extrait {len(names)} entrées")


def process_source(name: str, keep_archives: bool) -> None:
    src = SOURCES[name]
    out_dir = RAW / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in src["files"]:
        marker = out_dir / f".{spec['name']}.extracted"
        if marker.exists():
            print(f"[{name}] {spec['name']} : déjà extrait, rien à faire")
            continue
        archive = out_dir / spec["name"]
        if archive_ok(archive, spec):
            print(f"[{name}] {spec['name']} : archive déjà présente")
        else:
            print(f"[{name}] {spec['name']} : téléchargement (~{spec['size'] / 1e6:.0f} Mo)")
            fetch(spec, archive)
            if not archive_ok(archive, spec):
                sys.exit(f"[{name}] {spec['name']} : archive invalide (md5/zip), abandon")
        print(f"[{name}] {spec['name']} : extraction")
        extract(archive, out_dir, spec["members"])
        marker.touch()
        if not keep_archives:
            archive.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--keep-archives", action="store_true", help="ne pas supprimer les .zip après extraction")
    parser.add_argument("--list", action="store_true", help="affiche les sources et tailles, sans télécharger")
    args = parser.parse_args()

    if args.list:
        total = 0
        for name in args.sources:
            size = sum(f["size"] for f in SOURCES[name]["files"])
            total += size
            print(f"{name:10s} {size / 1e9:6.2f} Go  {SOURCES[name]['license']}")
        print(f"{'total':10s} {total / 1e9:6.2f} Go")
        return
    for name in args.sources:
        process_source(name, args.keep_archives)


if __name__ == "__main__":
    main()
