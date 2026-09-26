"""Read/write experiment result files as plain JSON or gzip-compressed JSON.

Convention for NEW runs: write with save_result(path.with_suffix('.json.gz'), obj). Full weight
traces compress roughly 5-10x, which matters for working-tree size and OneDrive sync (git already
compresses objects, so this does not shrink .git). load_result reads either format, so all the
existing plain-.json data keeps working unchanged; existing files are deliberately not converted
(rewriting them would only add duplicate blobs to git history).
"""
import gzip
import json
from pathlib import Path


def save_result(path, obj):
    """Write obj as JSON; gzip-compressed when the path ends in .gz."""
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(obj, f)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f)


def load_result(path):
    """Read a result written by save_result, or any plain .json result. If the exact path does
    not exist but its .gz sibling does (or the reverse), that one is read, so callers can keep
    using the original filename."""
    path = Path(path)
    if not path.exists():
        alt = Path(str(path) + '.gz') if path.suffix != '.gz' else path.with_suffix('')
        if alt.exists():
            path = alt
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    with open(path, encoding='utf-8') as f:
        return json.load(f)
