from pathlib import Path
import hashlib


def verify_sha256(path:Path, expected:str) -> bool:
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest() == expected