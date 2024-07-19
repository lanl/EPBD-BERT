from typing import Dict

import torch
from torch.utils.data import Dataset
import transformers
import pandas as pd
import numpy as np

from ..utility import pickle_utils

# from DNABERT2 Table 8, the following are the tfbs with data index
# Ch12Nrf2Iggrab: 0
# Ch12Znf384hpa004051Iggrab: 1
# MelJundIggrab: 2
# MelMafkDm2p5dStd: 3
# MelNelfeIggrab:4


class MouseSequenceEPBDDataset(Dataset):
    def __init__(self, index: int, data_type: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        assert index in list(range(5)), f"index must be in [0, 1, 2, 3, 4]"
        assert data_type in ["train", "dev", "test"], f"data_type must be in data_type: ['train', 'dev', 'test']"
        self.index, self.data_type = index, data_type

        data_path = f"../data/mouse_tfbs/mouse/{index}/{data_type}.csv"
        self.data_df = pd.read_csv(data_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data_df.shape[0]

    def _get_epbd_features(self, fname):
        feat_path = "/lustre/scratch4/turquoise/akabir/mouse_tfbs_epbd_features/id_seqs/"
        fpath = feat_path + fname
        data = pickle_utils.load(fpath)

        # coord and flip features
        coord = np.expand_dims(data["coord"], axis=0)
        flips = data["flip_verbose"] if data["flip_verbose"].shape[0] == 5 else np.transpose(data["flip_verbose"])
        # print(coord.shape, flips.shape)  # (1, 101) (5, 101)
        epbd_features = np.concatenate([coord, flips], axis=0) / 80000
        epbd_features = torch.tensor(epbd_features, dtype=torch.float32)
        # print(epbd_features.shape)  # [6, 101]
        return epbd_features

    def _tokenize_seq(self, seq: str):
        toked = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        )
        # print(toked)
        return toked["input_ids"].squeeze(0)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        x = self.data_df.loc[i]
        seq, label = x["sequence"], torch.tensor(int(x["label"]), dtype=torch.float32).unsqueeze(0)
        # print(seq, label)
        input_ids = self._tokenize_seq(seq)
        epbd_features = self._get_epbd_features(f"{self.index}_{self.data_type}_{i}.pkl")

        return dict(input_ids=input_ids, epbd_features=epbd_features, labels=label)


# tokenizer = transformers.AutoTokenizer.from_pretrained("resources/DNABERT-2-117M/", trust_remote_code=True, cache_dir="resources/cache/")
# ds = MouseSequenceEPBDDataset(index=0, data_type="train", tokenizer=tokenizer)
# print(ds.__len__())
# print(ds.__getitem__(0))

# run instructions
# no python package import needed
# from tf_dna_binding/epbd-bert> python -m epbd_bert.mouse_tfbs.dataset
