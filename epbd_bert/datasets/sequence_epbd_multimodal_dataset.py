import numpy as np
import torch
import transformers

import epbd_bert.utility.pickle_utils as pickle_utils
from epbd_bert.datasets.sequence_epbd_dataset import SequenceEPBDDataset


class SequenceEPBDMultiModalDataset(SequenceEPBDDataset):
    """Dataset for multi-modal transformer"""

    def __init__(self, data_path: str, pydnaepbd_features_path: str, tokenizer: transformers.PreTrainedTokenizer, home_dir=""):
        super().__init__(data_path, pydnaepbd_features_path, tokenizer, home_dir)

    def _get_epbd_features(self, fname):
        fpath = self.feat_path + fname
        data = pickle_utils.load(fpath)

        # coord and flip features
        coord = np.expand_dims(data["coord"], axis=0)
        flips = data["flip_verbose"] if data["flip_verbose"].shape[0] == 5 else np.transpose(data["flip_verbose"])
        # print(coord.shape, flips.shape)  # (1, 200) (5, 200)
        epbd_features = np.concatenate([coord, flips], axis=0) / 80000
        epbd_features = torch.tensor(epbd_features, dtype=torch.float32)
        # print(epbd_features.shape) # [6, 200]
        return epbd_features


# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     "resources/DNABERT-2-117M/",
#     trust_remote_code=True,
#     cache_dir="resources/cache/",
# )
# ds = SequenceEPBDMultiModalDataset(
#     data_path="resources/train_val_test/peaks_with_labels_test.tsv.gz",
#     pydnaepbd_features_path="resources/pydnaepbd_things/coord_flips/id_seqs/",  # ../data, resources
#     tokenizer=tokenizer,
# )
# print(ds.__len__())
# print(ds.__getitem__(100))

# to run
# python -m epbd_bert.datasets.sequence_epbd_multimodal_dataset
