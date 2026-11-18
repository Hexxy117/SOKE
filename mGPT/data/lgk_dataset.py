"""LGK dataset loader without body-model asset imports."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .lgk_contract import read_contract, validate_codes


class LGKDataset(Dataset):
    def __init__(self, data_root, split, mean, std, stage='lm_pretrain',
                 code_path='TOKENS_lgk', task_path=None, expected_fps=25,
                 **kwargs):
        self.root = Path(data_root)
        contract = read_contract(self.root, expected_fps)
        self.samples = [a for a in contract['samples'] if a['split'] == split]
        if not self.samples:
            raise ValueError(f'LGK {split} split is empty')
        self.tokens = 'lm' in stage and code_path is not None
        if self.tokens:
            metadata = validate_codes(self.root, contract, code_path, kwargs.get('normalization'))
            if kwargs.get('tokenizer_sha256') and metadata['checkpoint_sha256'] != kwargs['tokenizer_sha256']:
                raise ValueError('Motion tokens were generated with a different DETO checkpoint')
        self.code_path = code_path
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.stage = stage
        self.task = {'class': 't2m', 'input': ['<Caption_Placeholder>'],
                     'output': ['<Motion_Placeholder>']}
        if task_path:
            self.task = json.loads(Path(task_path).read_text(encoding='utf-8'))['Text-to-Motion']['t2m']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        a = self.samples[index]
        if self.tokens:
            motion = np.load(self.root / self.code_path / 'lgk' / (a['name'] + '.npy'), allow_pickle=False)[0]
            tensor = torch.from_numpy(motion.copy()).long()
        else:
            motion = np.load(self.root / a['motion'], allow_pickle=False)
            tensor = torch.from_numpy((motion - self.mean) / (self.std + 1e-10)).float()
        text = a['name'] if self.stage == 'token' else a['text']
        return text, tensor, len(tensor), a['name'], None, None, None, [a['text']], self.task, 'lgk'


