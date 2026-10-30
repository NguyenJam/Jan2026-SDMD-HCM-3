"""Download or verify the exact CMUdict sources, using only the standard library."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def load_lock(root=ROOT):
    return json.loads((root / "source_lock.json").read_text(encoding="utf-8"))


def verify_sources(root=ROOT):
    """Fail on missing or modified source bytes before interpreting the data."""
    records = []
    for item in load_lock(root)["files"]:
        path = root / "source" / item["name"]
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Checksum mismatch for {item['name']}: {actual}")
        records.append({**item, "bytes": len(content), "verified": True})
    return records


def fetch(root=ROOT):
    """Never overwrite an existing file that differs from the locked source."""
    directory = root / "source"
    directory.mkdir(parents=True, exist_ok=True)
    for item in load_lock(root)["files"]:
        destination = directory / item["name"]
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Existing source differs from lock: {destination}")
            continue
        request = Request(item["url"], headers={"User-Agent": "COS40007-data-profile/1.0"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Downloaded bytes do not match the lock: {item['name']}")
        # Write only after validation; avoid partial files if a download fails.
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        try:
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return verify_sources(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Do not use the network")
    args = parser.parse_args()
    try:
        records = verify_sources() if args.verify_only else fetch()
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Source verification failed: {exc}\n")
    for record in records:
        print(f"OK {record['name']} {record['bytes']} bytes SHA256={record['sha256']}")
