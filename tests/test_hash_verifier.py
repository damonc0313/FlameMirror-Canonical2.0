import pathlib
import sys

# Ensure project root on path for imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from hash_verifier import compute_sha256, verify_hashes


def test_compute_sha256():
    path = pathlib.Path('tests/data/sample.txt')
    expected = 'a53d426a3902b97e8e4056c0de26c5769df7f1784a411d0c70f123c59715940f'
    assert compute_sha256(path) == expected


def test_verify_hashes_success():
    csv_path = pathlib.Path('tests/data/sample_hashes.csv')
    result = verify_hashes(csv_path, base_path=csv_path.parent)
    assert result == {'sample.txt': True}
