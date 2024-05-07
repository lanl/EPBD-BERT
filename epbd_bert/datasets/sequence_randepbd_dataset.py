import torch
import transformers

from epbd_bert.datasets.sequence_epbd_dataset import SequenceEPBDDataset


class SequenceRandEPBDDataset(SequenceEPBDDataset):
    def __init__(self, data_path: str, pydnaepbd_features_path: str, tokenizer: transformers.PreTrainedTokenizer, home_dir=""):
        super().__init__(data_path, pydnaepbd_features_path="", tokenizer=tokenizer, home_dir=home_dir)

    def _get_epbd_features(self, fname):
        return torch.rand(1200, dtype=torch.float32)


# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     "resources/DNABERT-2-117M/",
#     trust_remote_code=True,
#     cache_dir="resources/cache/",
# )
# ds = SequenceRandEPBDDataset(
#     data_path="resources/train_val_test/peaks_with_labels_test.tsv.gz",
#     pydnaepbd_features_path="",
#     tokenizer=tokenizer,
# )

# print(ds.__len__())
# print(ds.__getitem__(100))

# to run
# python -m epbd_bert.datasets.sequence_randepbd_dataset
