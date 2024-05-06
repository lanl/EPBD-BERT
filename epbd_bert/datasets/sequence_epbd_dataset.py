from typing import Dict
import numpy as np
import torch
import transformers

import epbd_bert.utility.pickle_utils as pickle_utils
from epbd_bert.datasets.sequence_dataset import SequenceDataset

# from epbd_bert.path_configs import pydnaepbd_features_path


class SequenceEPBDDataset(SequenceDataset):
    """Supervised fine-tuning from seq and epbd features"""

    def __init__(self, data_path: str, pydnaepbd_features_path: str, tokenizer: transformers.PreTrainedTokenizer, home_dir=""):
        super().__init__(data_path, tokenizer, home_dir)
        self.feat_path = home_dir + pydnaepbd_features_path

    def _get_epbd_features(self, fname):
        fpath = self.feat_path + fname
        data = pickle_utils.load(fpath)

        # coord and flip features
        concatenated_data = np.concatenate([data["coord"], data["flip_verbose"].flatten()]) / 80000

        # only coords features
        # concatenated_data = data["coord"] / 80000
        epbd_features = torch.tensor(concatenated_data, dtype=torch.float32)
        # print(epbd_features.shape) #[1200]
        return epbd_features

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        chrom, start, end, labels = self._get_seq_position_and_labels(i)
        seq_id = f"{chrom}_{str(start)}_{str(end)}"
        # tokenize seq
        input_ids = self._tokenize_seq(seq_id)
        # label generation
        labels = self._get_label_vector(labels)
        # epbd features
        epbd_features = self._get_epbd_features(f"{seq_id}.pkl")

        return dict(input_ids=input_ids, epbd_features=epbd_features, labels=labels)
