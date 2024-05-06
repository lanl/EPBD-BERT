import transformers
from epbd_bert.datasets.sequence_epbd_multimodal_dataset import (
    SequenceEPBDMultiModalDataset,
)


class SequenceEPBDMultiModalLabelSpecificDataset(SequenceEPBDMultiModalDataset):
    """Dataset for multi-modal transformer"""

    def __init__(
        self,
        data_path: str,
        pydnaepbd_features_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        label="wgEncodeAwgTfbsBroadDnd41CtcfUniPk",
        home_dir="",
    ):
        super().__init__(data_path, pydnaepbd_features_path, tokenizer, home_dir)
        self.data_df = self.data_df[self.data_df["labels"].apply(lambda x: label in x)]
        self.data_df.reset_index(drop=True, inplace=True)


# data_path = "data/train_val_test/peaks_with_labels_test.tsv.gz"
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     "data/dnabert2_pretrained/DNABERT-2-117M/",
#     trust_remote_code=True,
#     cache_dir="data/dnabert2_pretrained/cache/",
# )
# d = SequenceEPBDMultiModalLabelSpecificDataset(
#     data_path, tokenizer, label="wgEncodeAwgTfbsBroadDnd41CtcfUniPk"
# )
# print(d.__len__())
# print(d.__getitem__(100))
