from typing import Dict

import transformers
import pandas as pd

import torch
from torch.utils.data import Dataset

import epbd_bert.utility.pickle_utils as pickle_utils


class SequenceDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, home_dir=""):
        super().__init__()
        data_path = home_dir + data_path
        # "data/train_val_test/peaks_with_labels_val.tsv.gz"
        labels_dict_path = home_dir + "resources/processed_data/peakfilename_index_dict.pkl"
        seqs_dict_path = home_dir + "resources/processed_data/seq_with_flanks_dict.pkl"
        # this may contain "N"s since the flanks are not cleaned but 200seq are cleaned
        self.tokenizer = tokenizer

        self.data_df = pd.read_csv(data_path, compression="gzip", sep="\t")
        self.labels_dict = pickle_utils.load(labels_dict_path)
        self.seq_dict = pickle_utils.load(seqs_dict_path)
        # print(self.data_df.shape, len(self.labels_dict), len(self.seq_dict))

        self.num_labels = len(self.labels_dict)

    def _get_label_vector(self, labels: str):
        y = torch.zeros(len(self.labels_dict), dtype=torch.float32)

        for l in labels.split(","):
            l = l.strip()
            y[self.labels_dict[l]] = 1

        # print(y)
        return y

    def _get_seq_position_and_labels(self, i: int):
        x = self.data_df.loc[i]
        chrom, start, end, labels = (
            x["chrom"],
            int(x["start"]),
            int(x["end"]),
            x["labels"],
        )
        # chrom, start, end, labels = (
        #     "chr8",
        #     67025400,
        #     67025600,
        #     "wgEncodeAwgTfbsSydhK562Brf1UniPk",
        # )
        # print(chrom, start, end, labels)

        return chrom, start, end, labels

    def _tokenize_seq(self, seq_id: str):
        # example seq and labels to debug
        # seq = "NCCTTGCTCCTGTCTCAGGACACAGAGCCATGGACGACCACCCTTGCTCCTGTCTCAGG"
        # labels = "wgEncodeAwgTfbsSydhH1hescCebpbIggrabUniPk, wgEncodeAwgTfbsSydhNb4MaxUniPk"

        seq = self.seq_dict[seq_id]
        # print(len(seq), seq)

        toked = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        )
        # print(toked)
        return toked["input_ids"].squeeze(0)

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        chrom, start, end, labels = self._get_seq_position_and_labels(i)
        seq_id = f"{chrom}_{str(start)}_{str(end)}"
        # tokenize seq
        input_ids = self._tokenize_seq(seq_id)
        # label generation
        labels = self._get_label_vector(labels)

        # print(input_ids.shape, input_ids.dtype, labels.shape, labels.dtype)
        return dict(input_ids=input_ids, labels=labels)


# data_path = "resources/train_val_test/peaks_with_labels_val.tsv.gz"
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     "resources/DNABERT-2-117M/",
#     trust_remote_code=True,
#     cache_dir="resources/cache/",
# )
# ds = SequenceDataset(data_path, tokenizer)
# print(ds.__len__())
# print(ds.__getitem__(100))

# to run
# python -m epbd_bert.datasets.sequence_dataset
