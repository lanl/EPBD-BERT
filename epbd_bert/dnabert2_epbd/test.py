import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics

from epbd_bert.datasets.sequence_epbd_dataset import SequenceEPBDDataset
from epbd_bert.datasets.data_collators import SeqLabelEPBDDataCollator
from epbd_bert.dnabert2_epbd.model import Dnabert2EPBDModel
from epbd_bert.utility.dnabert2 import get_dnabert2_tokenizer
import epbd_bert.utility.pickle_utils as pickle_utils


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
    data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)

    test_dataset = SequenceEPBDDataset(
        data_path="data/train_val_test/peaks_with_labels_test.tsv.gz",
        tokenizer=tokenizer,
    )
    test_dl = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        shuffle=False,
        pin_memory=False,
        batch_size=64,
        num_workers=10,
    )
    print(test_dataset.__len__(), len(test_dl))

    def get_predictions(dl: DataLoader, saved_preds_path: str, compute_again=False):
        if os.path.exists(saved_preds_path) and not compute_again:
            preds_and_targets_dict = pickle_utils.load_pickle(saved_preds_path)
            print(
                preds_and_targets_dict["preds"].shape,
                preds_and_targets_dict["targets"].shape,
            )
            return preds_and_targets_dict

        model = Dnabert2EPBDModel.load_from_checkpoint(
            "dnabert2_epbd/lightning_logs/version_0/checkpoints/epoch=4-step=44508.ckpt"
        )
        # model.to(device)
        model.eval()

        all_preds, all_targets = [], []
        for i, batch in enumerate(dl):
            # if i < 1265:
            #     continue
            x = {
                "input_ids": batch["input_ids"].to(device),
                "epbd_features": batch["epbd_features"].to(device),
                "labels": batch["labels"],
                "attention_mask": batch["attention_mask"].to(device),
            }
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
        print(
            preds_and_targets_dict["preds"].shape,
            preds_and_targets_dict["targets"].shape,
        )
        pickle_utils.save_as_pickle(preds_and_targets_dict, saved_preds_path)
        return preds_and_targets_dict

    preds_and_targets_dict = get_predictions(
        test_dl,
        saved_preds_path="dnabert2_epbd/backups/pred_and_targets_dict_ontest_v0.pkl",
        compute_again=False,
    )

    # overall auc-roc
    auc_roc = metrics.roc_auc_score(
        preds_and_targets_dict["targets"],
        preds_and_targets_dict["preds"],
        average="micro",
    )
    print("overall auc-roc: ", auc_roc)

    # these codes are taken from tbinet
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

    get_aurocs_and_auprs(
        preds_and_targets_dict["preds"],
        preds_and_targets_dict["targets"],
        True,
    )
