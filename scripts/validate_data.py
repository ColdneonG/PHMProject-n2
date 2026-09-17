"""Validate files and manifests for the fixed study dataset."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def validate(root):
    with (root / 'split_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    counts = Counter(r['new_split'] for r in records)
    assert counts == {'train': 4593, 'val': 588, 'test': 1354}, counts
    groups, hashes = defaultdict(set), defaultdict(set)
    expected_images = set()
    expected_masks = set()
    for record in records:
        split = record['new_split']
        physical = re.match(r'(video\d+)', record['video'], flags=re.I).group(1)
        assert physical == record['group'], record
        groups[physical].add(split)
        for field in ['dst_image', 'dst_mask']:
            rel = Path(record[field].replace('\\', '/'))
            if rel.parts[0] == 'data':
                rel = Path(*rel.parts[1:])
            assert not rel.is_absolute() and '..' not in rel.parts
            path = root / rel
            assert path.is_file(), str(path)
            (expected_images if field == 'dst_image' else expected_masks).add(rel.as_posix())
            if field == 'dst_image':
                hashes[hashlib.sha256(path.read_bytes()).hexdigest()].add(split)
    assert not {k: list(v) for k, v in groups.items() if len(v) > 1}, 'Physical video crosses splits'
    cross_split = sum(len(v) > 1 for v in hashes.values())
    assert cross_split == 0, f'{cross_split} identical image byte hashes cross splits'
    for folder, expected in [('imgs', expected_images), ('masks', expected_masks)]:
        actual = {p.relative_to(root).as_posix() for p in (root / folder).rglob('*') if p.is_file()}
        assert actual == expected, {'folder': folder, 'extra': len(actual-expected), 'missing': len(expected-actual)}
    with (root / 'condition_image_level.csv').open(encoding='utf-8-sig', newline='') as f:
        conditions = {(r['split'], r['video'], r['frame']): r['condition'] for r in csv.DictReader(f)}
    assert len(conditions) == len(records)
    for r in records:
        assert conditions[(r['new_split'], r['video'], r['new_frame'])] == r['condition']
    print(json.dumps({'frames': dict(counts), 'physical_videos': len(groups),
                      'cross_split_identical_image_hashes': cross_split,
                      'conditions_verified': len(conditions), 'status': 'passed'}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    validate(parser.parse_args().root.resolve())
