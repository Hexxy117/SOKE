"""Lightweight LGK dataset validation, shared by loaders and CLI checks."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_contract(root, expected_fps=None, require_all_splits=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    if manifest['version'] != 1 or manifest['representation'] != 'soke_smplx_133_raw':
        raise ValueError('Unsupported LGK dataset format')
    if expected_fps is not None and abs(manifest['fps'] - expected_fps) > 1e-6:
        raise ValueError('Dataset/config frame rate mismatch')
    names, recordings = set(), {}
    for s in ('train', 'val', 'test'):
        if require_all_splits and not manifest['splits'][s]:
            raise ValueError(f'Empty {s} split: assign a separate recording')
    for ann in manifest['samples']:
        name, split = ann['name'], ann['split']
        if name in names or split not in ('train', 'val', 'test'):
            raise ValueError(f'Duplicate sample or invalid split: {name}')
        names.add(name)
        rid = ann['recording_id']
        if rid in recordings and recordings[rid] != split:
            raise ValueError(f'Recording leakage across splits: {rid}')
        recordings[rid] = split
        if ann['fps'] != manifest['fps'] or name not in manifest['splits'][split]:
            raise ValueError(f'Inconsistent sample metadata: {name}')
        path = (root / ann['motion']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Motion path escapes dataset root')
        motion = np.load(path, allow_pickle=False)
        if (motion.shape != (ann['frames'], 133) or not np.isfinite(motion).all()
                or not 40 <= len(motion) <= 400 or len(motion) % 4):
            raise ValueError(f'Invalid motion: {name}')
        if sha256(path) != ann['sha256'] or not ann['text'].strip():
            raise ValueError(f'Modified motion or empty text: {name}')
    listed = [n for split in manifest['splits'].values() for n in split]
    if len(listed) != len(set(listed)) or set(listed) != names:
        raise ValueError('Split lists differ from sample inventory')
    return manifest


def validate_codes(root, manifest, code_path='TOKENS_lgk', normalization=None):
    root = Path(root)
    meta = json.loads((root / code_path / 'metadata.json').read_text(encoding='utf-8'))
    if meta['dataset_sha256'] != sha256(root / 'dataset.json'):
        raise ValueError('Dataset changed after token generation; regenerate tokens')
    if normalization is not None and meta['normalization'] != normalization:
        raise ValueError('Token and model normalization differ; regenerate tokens')
    for ann in manifest['samples']:
        path = root / code_path / 'lgk' / (ann['name'] + '.npy')
        codes = np.load(path, allow_pickle=False)
        if codes.shape != (1, ann['frames'] // 4, 3) or not np.issubdtype(codes.dtype, np.integer):
            raise ValueError(f'Invalid token shape/dtype: {path}')
        for i, maximum in enumerate((96, 192, 192)):
            if (codes[..., i] < 0).any() or (codes[..., i] >= maximum).any():
                raise ValueError(f'Out-of-codebook token: {path}')
        if sha256(path) != meta['codes_sha256'][ann['name']]:
            raise ValueError(f'Modified token file: {path}')
    return meta
