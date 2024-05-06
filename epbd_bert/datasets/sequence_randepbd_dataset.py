import torch
import transformers

from epbd_bert.datasets.sequence_epbd_dataset import SequenceEPBDDataset


class SequenceRandEPBDDataset(SequenceEPBDDataset):
    def __init__(
        self, data_path: str, pydnaepbd_features_path:str, tokenizer: transformers.PreTrainedTokenizer, home_dir=""
    ):
        super().__init__(data_path, pydnaepbd_features_path="", tokenizer, home_dir)

    def _get_epbd_features(self, fname):
        return torch.rand(1200, dtype=torch.float32)
