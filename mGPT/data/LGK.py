"""LGK data module; uses the original SOKE geometry and normalization."""
from .H2S import H2SDataModule
from .lgk_contract import sha256
from .lgk_dataset import LGKDataset


class LGKDataModule(H2SDataModule):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.Dataset = LGKDataset
        self.DatasetEval = LGKDataset
        self.hparams.expected_fps = float(cfg.DATASET.H2S.FRAME_RATE)
        self.hparams.normalization = {
            'mean_sha256': sha256(cfg.DATASET.H2S.MEAN_PATH),
            'std_sha256': sha256(cfg.DATASET.H2S.STD_PATH)}
        self.hparams.tokenizer_sha256 = sha256(cfg.TRAIN.PRETRAINED_VAE)
        self.hparams.task_path = cfg.DATASET.TASK_PATH
        self.hparams.code_path = cfg.DATASET.CODE_PATH
