"""Utility for verifying file integrity using SHA-256 hashes.

This module reads a CSV mapping file names to their expected SHA-256
hashes and validates the hashes of the corresponding files. It can be
used as a library or executed as a script.

Example:
    python hash_verifier.py FlameMirror_Individual_File_Hashes.csv
"""

from __future__ import annotations

import csv
import hashlib
import pathlib
from typing import Dict


def compute_sha256(path: pathlib.Path) -> str:
    """Compute SHA-256 hash of a file.

    Parameters
    ----------
    path: pathlib.Path
        Path to the file whose hash will be computed.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest of the file's contents.
    """
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha.update(chunk)
    return sha.hexdigest()


def verify_hashes(csv_path: pathlib.Path, base_path: pathlib.Path | None = None) -> Dict[str, bool]:
    """Verify files listed in a CSV against expected SHA-256 hashes.

    Parameters
    ----------
    csv_path: pathlib.Path
        Path to the CSV file containing ``File`` and ``SHA-256`` columns.
    base_path: pathlib.Path, optional
        Base directory to resolve file paths. Defaults to the directory
        containing ``csv_path``.

    Returns
    -------
    Dict[str, bool]
        Mapping of file names to verification result ``True`` if the
        computed hash matches the expected hash, ``False`` otherwise.
    """
    if base_path is None:
        base_path = csv_path.parent

    results: Dict[str, bool] = {}
    with csv_path.open(newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            filename = row.get('File') or row.get('file')
            expected = (row.get('SHA-256') or row.get('sha256') or '').strip()
            if not filename:
                continue
            file_path = base_path / filename
            if not file_path.exists():
                results[filename] = False
                continue
            actual = compute_sha256(file_path)
            results[filename] = actual.lower() == expected.lower()
    return results


def main():
    """Main entry point for the hash verifier script."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Verify file hashes from a CSV")
    parser.add_argument("csv", type=pathlib.Path, help="Path to CSV file with expected hashes")
    parser.add_argument("--base", type=pathlib.Path, default=None, help="Base directory for files")
    args = parser.parse_args()

    result = verify_hashes(args.csv, args.base)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
