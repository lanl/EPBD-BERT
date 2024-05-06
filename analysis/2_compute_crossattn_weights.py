import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import utility.pickle_utils as pickle_utils
from datasets.data_collators import SeqLabelEPBDDataCollator
from analysis.x_model import EPBDDnabert2ModelForAnalysis

# loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ckpt_path = "dnabert2_epbd_crossattn/backups/v1/checkpoints/epoch=12-step=125151.ckpt"
ckpt_path = "analysis/best_model/epoch=9-step=255700.ckpt"
model = EPBDDnabert2ModelForAnalysis.load_from_checkpoint(ckpt_path)
model.to(device)
model.eval()
# print(model)


def compute_model_items(
    ds: Dataset,
    tokenizer,
    out_filepath: str,
    item_name="cross_attn_weights",
    max_seq=None,
    apply_avg=False,
):
    # max_seq=None will process all seqs
    # example ds
    # data_path = "data/train_val_test/peaks_with_labels_test.tsv.gz"
    # tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
    # ds = SequenceEPBDMultiModalLabelSpecificDataset(
    #     data_path, tokenizer, label="wgEncodeAwgTfbsSydhHepg2Mafksc477IggrabUniPk"
    # )

    if os.path.exists(out_filepath):
        return

    data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)
    dl = DataLoader(
        ds,
        collate_fn=data_collator,
        shuffle=False,
        pin_memory=False,
        batch_size=1,
        num_workers=1,
    )
    print("DS, DL:", ds.__len__(), len(dl))

    # running the model on the given ds
    items = []
    for i, batch in enumerate(dl):
        batch = {key: batch[key].to(device) for key in batch.keys()}
        outs = model.analysis_forward(batch)
        # print(outs.keys())

        if apply_avg:
            x = np.mean(outs[item_name], axis=0)
            # print(x.shape)
            items.append(x)
        else:
            items.append(outs[item_name])  # item_name can be the keys in the outs

        if i == max_seq:
            # max_seq=None will process all seqs
            break
        if i % 100 == 0:
            print(i)

        # break

    pickle_utils.save(items, out_filepath)  # list of items


# common to all usecases
from utility.dnabert2 import get_dnabert2_tokenizer

tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)

# -----------example: cross-attn-weights for all labels or 1 label
# from datasets.sequence_epbd_multimodal_labelspecific_dataset import (
#     SequenceEPBDMultiModalLabelSpecificDataset,
# )

# data_path = "data/train_val_test/peaks_with_labels_test.tsv.gz"
# labels_dict = pickle_utils.load("data/processed/peakfilename_index_dict.pkl")
# for label, label_id in labels_dict.items():
#     # label = "wgEncodeAwgTfbsSydhHelas3Bdp1UniPk"
#     print(label_id, label)
#     ds = SequenceEPBDMultiModalLabelSpecificDataset(data_path, tokenizer, label)
#     out_filepath = (
#         f"analysis/weights/690_label_specific/{label}_crossattn_weights_list.pkl"
#     )
#     compute_model_items(
#         ds, tokenizer, out_filepath, item_name="cross_attn_weights", max_seq=None
#     )
#     # break


# -----------example: DNABERT2 did the analysis for the following two tf-dna binding events
# wgEncodeAwgTfbsUtaA549CtcfUniPk (wgEncodeEH002078), wgEncodeAwgTfbsSydhHelas3Brg1IggmusUniPk (wgEncodeEH000781)
# from datasets.sequence_epbd_multimodal_labelspecific_dataset import (
#     SequenceEPBDMultiModalLabelSpecificDataset,
# )

# data_path = "data/train_val_test/peaks_with_labels_test.tsv.gz"
# labels_dict = pickle_utils.load("data/processed/peakfilename_index_dict.pkl")
# label = "wgEncodeAwgTfbsSydhHelas3Brg1IggmusUniPk"
# ds = SequenceEPBDMultiModalLabelSpecificDataset(data_path, tokenizer, label)
# out_filepath = f"analysis/weights/690_label_specific/{label}_crossattn_weights_list.pkl"
# compute_model_items(ds, tokenizer, out_filepath, max_seq=None)


# -----------example: test seq cross-attn weights with real EPBD features
# from datasets.sequence_epbd_multimodal_dataset import SequenceEPBDMultiModalDataset

# ds = SequenceEPBDMultiModalDataset(
#     "data/train_val_test/peaks_with_labels_test.tsv.gz", tokenizer
# )
# out_filepath = f"analysis/weights/10k_test_seq_crossattn_weights_epbd_list.pkl"
# compute_model_items(ds, tokenizer, out_filepath, max_seq=10000)
# out_filepath = f"analysis/weights/all_test_seq_crossattn_weights_epbd_list.pkl"
# compute_model_items(ds, tokenizer, out_filepath, max_seq=None)

# example 3: test seq cross-attn weights with random EPBD features
# from datasets.sequence_randepbd_multimodal_dataset import (
#     SequenceRandEPBDMultiModalDataset,
# )

# ds = SequenceRandEPBDMultiModalDataset(
#     "data/train_val_test/peaks_with_labels_test.tsv.gz", tokenizer
# )
# out_filepath = f"analysis/weights/all_test_seq_crossattn_weights_randepbd_list.pkl"
# compute_model_items(ds, tokenizer, out_filepath, max_seq=None)


# -----------example: saving epbd features after the convolution operation
# from datasets.sequence_epbd_multimodal_labelspecific_dataset import (
#     SequenceEPBDMultiModalLabelSpecificDataset,
# )

# data_path = "data/train_val_test/peaks_with_labels_test.tsv.gz"
# labels_dict = pickle_utils.load("data/processed/peakfilename_index_dict.pkl")
# for label, label_id in labels_dict.items():
#     label = "wgEncodeAwgTfbsSydhHepg2Mafksc477IggrabUniPk"
#     print(label_id, label)
#     ds = SequenceEPBDMultiModalLabelSpecificDataset(data_path, tokenizer, label)
#     out_filepath = f"analysis/conved_epbd_features_list/{label}.pkl"
#     compute_model_items(
#         ds, tokenizer, out_filepath, item_name="avg_epbd_features", max_seq=None
#     )
#     break


# -----------example: test/val all seq average cross-attn weights with real EPBD features
from datasets.sequence_epbd_multimodal_dataset import SequenceEPBDMultiModalDataset

split = "val"  # val, test
ds = SequenceEPBDMultiModalDataset(
    f"data/train_val_test/peaks_with_labels_{split}.tsv.gz", tokenizer
)
out_filepath = f"analysis/weights/all_{split}_seq_avg_crossattn_weights_list.pkl"
compute_model_items(
    ds,
    tokenizer,
    out_filepath,
    max_seq=None,
    item_name="cross_attn_weights",
    apply_avg=True,
)
