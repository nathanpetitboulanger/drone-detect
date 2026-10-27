"""Télécharge les polices libres (SIL OFL 1.1) utilisées par la vidéo de présentation.

Polices variables du dépôt google/fonts, vérifiées par SHA-256 :
  - Inter (titres et texte) → video/fonts/Inter-Variable.ttf
  - JetBrains Mono (chiffres, valeurs « instrument ») → video/fonts/JetBrainsMono-Variable.ttf
  + les licences OFL correspondantes.

Idempotent : ne retélécharge pas un fichier déjà présent avec la bonne empreinte.

Usage (depuis la racine du dépôt) :
    .venv/bin/python video/fetch_fonts.py
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "video" / "fonts"
BASE = "https://github.com/google/fonts/raw/main/ofl"

# nom local -> (URL, SHA-256 ou None pour les licences)
FILES = {
    "Inter-Variable.ttf": (
        f"{BASE}/inter/Inter%5Bopsz,wght%5D.ttf",
        "29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031",
    ),
    "JetBrainsMono-Variable.ttf": (
        f"{BASE}/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
        "48715a42ec242c21e9f02692891e147d022299a52e48d5e413e1a942193ffeda",
    ),
    "OFL-Inter.txt": (f"{BASE}/inter/OFL.txt", None),
    "OFL-JetBrainsMono.txt": (f"{BASE}/jetbrainsmono/OFL.txt", None),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    FONTS.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in FILES.items():
        dst = FONTS / name
        if dst.exists() and (digest is None or sha256(dst) == digest):
            print(f"[fonts] {name} : déjà présent")
            continue
        print(f"[fonts] {name} ← {url}")
        with urllib.request.urlopen(url, timeout=60) as r:
            dst.write_bytes(r.read())
        if digest is not None and sha256(dst) != digest:
            print(f"[fonts] attention : empreinte inattendue pour {name} (police mise à jour en amont ?)")


if __name__ == "__main__":
    main()
