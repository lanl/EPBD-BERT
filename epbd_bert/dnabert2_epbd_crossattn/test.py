import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics

from epbd_bert.datasets.sequence_epbd_multimodal_dataset import (
    SequenceEPBDMultiModalDataset,
)
from epbd_bert.datasets.data_collators import SeqLabelEPBDDataCollator
from epbd_bert.dnabert2_epbd_crossattn.model import EPBDDnabert2Model
from epbd_bert.utility.dnabert2 import get_dnabert2_tokenizer
import epbd_bert.utility.pickle_utils as pickle_utils


def compute_predictions(model, dl: DataLoader, output_preds_path: str, compute_again=False):
    if os.path.exists(output_preds_path) and not compute_again:
        preds_and_targets_dict = pickle_utils.load(output_preds_path)
        print(preds_and_targets_dict["preds"].shape, preds_and_targets_dict["targets"].shape)
        return preds_and_targets_dict

    all_preds, all_targets = [], []
    for i, batch in enumerate(dl):
        x = {key: batch[key].to(device) for key in batch.keys()}
        del batch
        logits, targets = model(x)
        logits, targets = logits.detach().cpu(), targets.detach().cpu()
        probs = F.sigmoid(logits)

        probs, targets = probs.numpy(), targets.numpy()
        print(i, probs.shape, targets.shape)

        all_preds.append(probs)
        all_targets.append(targets)

        # if i == 0:
        #     break

    # accumulating all predictions and target vectors
    all_preds, all_targets = np.vstack(all_preds), np.vstack(all_targets)
    preds_and_targets_dict = {"preds": all_preds, "targets": all_targets}
    print(preds_and_targets_dict["preds"].shape, preds_and_targets_dict["targets"].shape)
    pickle_utils.save(preds_and_targets_dict, output_preds_path)
    return preds_and_targets_dict


if __name__ == "__main__":
    tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
    data_collator = SeqLabelEPBDDataCollator(tokenizer.pad_token_id)
    test_dataset = SequenceEPBDMultiModalDataset(
        "data/train_val_test/peaks_with_labels_test.tsv.gz",
        pydnaepbd_features_path="data/pydnaepbd_things/coord_flips/id_seqs/",
        tokenizer=tokenizer,
    )
    test_dl = DataLoader(test_dataset, collate_fn=data_collator, shuffle=False, pin_memory=False, batch_size=64, num_workers=10)
    print(test_dataset.__len__(), len(test_dl))

    # version = "v1"  # v0, v1
    # checkpoint_path = f"dnabert2_epbd_crossattn/backups/{version}/checkpoints/epoch=12-step=125151.ckpt"  # epoch=29-step=72210-val_loss=0.058-val_auc_roc=0.939.ckpt"
    checkpoint_path = f"epbd-bert/resources/trained_weights/epbd_dnabert2_crossattn_best.ckpt"
    model = EPBDDnabert2Model.load_pretrained_model(checkpoint_path, mode="eval")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds_and_targets_dict = compute_predictions(
        model, test_dl, output_preds_path=f"epbd-bert/epbd_bert/dnabert2_epbd_crossattn/outputs/pred_and_targets_dict_ontest.pkl", compute_again=False
    )

    # overall auc-roc
    auc_roc = metrics.roc_auc_score(preds_and_targets_dict["targets"], preds_and_targets_dict["preds"], average="micro")
    print("overall auc-roc: ", auc_roc)

    # performance metrics computation
    def get_auroc(preds, obs):
        fpr, tpr, thresholds = metrics.roc_curve(obs, preds, drop_intermediate=False)
        auroc = metrics.auc(fpr, tpr)
        return auroc

    def get_aupr(preds, obs):
        precision, recall, thresholds = metrics.precision_recall_curve(obs, preds)
        aupr = metrics.auc(recall, precision)
        return aupr

    def get_aurocs_and_auprs(pred_probs: np.array, targets: np.array, verbose=False):
        auroc_list = []
        aupr_list = []
        for task in range(pred_probs.shape[1]):
            pred, obs = pred_probs[:, task], targets[:, task]
            auroc = round(get_auroc(pred, obs), 5)
            aupr = round(get_aupr(pred, obs), 5)
            auroc_list.append(auroc)
            aupr_list.append(aupr)

        if verbose:
            print("Averaged AUROC:", np.nanmean(auroc_list))
            print("Averaged AUPR:", np.nanmean(aupr_list))
            print(sorted(auroc_list, reverse=True))
            print(sorted(aupr_list, reverse=True))

    # computing and printing performance metrics
    get_aurocs_and_auprs(preds_and_targets_dict["preds"], preds_and_targets_dict["targets"], True)
