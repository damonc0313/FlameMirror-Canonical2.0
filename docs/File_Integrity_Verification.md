# File Integrity Verification

This repository includes a CSV file containing SHA-256 hashes for individual JSON files. The `hash_verifier.py` utility can be used to validate those hashes and check the integrity of downloaded artifacts.

## Usage

```bash
python hash_verifier.py FlameMirror_Individual_File_Hashes.csv
```

The script outputs a JSON object mapping each file name to a boolean indicating whether its hash matches the expected value. Missing files are reported as `false`.

## Library API

The module exposes two functions:

- `compute_sha256(path)` – compute the SHA-256 digest of a file.
- `verify_hashes(csv_path, base_path=None)` – validate files listed in the CSV. When `base_path` is not provided, file paths are resolved relative to the CSV location.

These functions can be imported and reused in other tooling to ensure repository integrity.
