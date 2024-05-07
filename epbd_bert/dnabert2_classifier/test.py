import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics

from epbd_bert.datasets.sequence_dataset import SequenceDataset
from epbd_bert.datasets.data_collators import SeqLabelDataCollator
from epbd_bert.utility.dnabert2 import get_dnabert2_tokenizer
from epbd_bert.dnabert2_classifier.model import DNABERT2Classifier
import epbd_bert.utility.pickle_utils as pickle_utils

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------- first 3 are necessary paths
    # model_ckpt_path = "dnabert2/backups/version_1/checkpoints/epoch=54-step=132385.ckpt"
    model_ckpt_path = "resources/trained_weights/dnabert2_classifier.ckpt"
    saved_preds_path = "outputs/dnabert2_pred_and_targets_dict_ontest.pkl"
    result_path = "outputs/dnabert2_result.tsv"
    test_data_path = "resources/train_val_test/peaks_with_labels_test.tsv.gz"
    labels_dict_path = "resources/processed_data/peakfilename_index_dict.pkl"

    tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
    data_collator = SeqLabelDataCollator(pad_token_id=tokenizer.pad)
    test_dataset = SequenceDataset(data_path=test_data_path, tokenizer=tokenizer)
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

        model = DNABERT2Classifier.load_from_checkpoint(model_ckpt_path)
        # model.to(device)
        model.eval()

        all_preds, all_targets = [], []
        for i, batch in enumerate(dl):
            # if i < 1265:
            #     continue
            x = {
                "input_ids": batch["input_ids"].to(device),
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

    preds_and_targets_dict = get_predictions(test_dl, saved_preds_path, compute_again=False)

    # overall auc-roc
    auc_roc = metrics.roc_auc_score(
        preds_and_targets_dict["targets"],
        preds_and_targets_dict["preds"],
        average="micro",
    )
    print("overall auc-roc: ", auc_roc)

    # loading labels-dict and labels-metadata which contains cell and antibody type
    label2index_dict = pickle_utils.load_pickle(labels_dict_path)
    index2label_dict = {i: label for label, i in label2index_dict.items()}
    # index2label_dict[0]
    peaks_metadata_df = pickle_utils.get_uniform_peaks_metadata()

    # computing auc-roc and auc-pr per cell/antibody type
    def get_auroc(preds, obs):
        fpr, tpr, ths = metrics.roc_curve(obs, preds, drop_intermediate=False)
        auroc = metrics.auc(fpr, tpr)
        return auroc

    def get_aupr(preds, obs):
        precision, recall, ths = metrics.precision_recall_curve(obs, preds)
        aupr = metrics.auc(recall, precision)
        return aupr

    def get_aurocs_and_auprs(pred_probs: np.array, targets: np.array, verbose=False):
        auroc_dict = {}
        aupr_dict = {}
        for task_id in range(pred_probs.shape[1]):
            pred, obs = pred_probs[:, task_id], targets[:, task_id]
            auroc = round(get_auroc(pred, obs), 5)
            aupr = round(get_aupr(pred, obs), 5)
            auroc_dict[index2label_dict[task_id]] = auroc
            aupr_dict[index2label_dict[task_id]] = aupr
        return auroc_dict, aupr_dict

    auroc_dict, aupr_dict = get_aurocs_and_auprs(
        preds_and_targets_dict["preds"],
        preds_and_targets_dict["targets"],
        True,
    )

    auroc_list = [v for k, v in auroc_dict.items()]
    aupr_list = [v for k, v in aupr_dict.items()]
    print("Averaged AUROC:", np.nanmean(auroc_list))
    print("Averaged AUPR:", np.nanmean(aupr_list))

    # post-processing computed auc-roc and -pr
    def get_df_from_dict(a_dict, value_col_name):
        df = pd.DataFrame.from_dict(a_dict, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={0: value_col_name}, inplace=True)
        return df

    auroc_df = get_df_from_dict(auroc_dict, value_col_name="auroc")
    aupr_df = get_df_from_dict(aupr_dict, value_col_name="aupr")
    performance_df = pd.merge(left=auroc_df, right=aupr_df, how="inner", on="index")
    # performance_df

    # extracting corresponding cell/antibody
    result_df = peaks_metadata_df[["tableName", "cell", "antibody"]].copy()
    result_df.rename(columns={"tableName": "index"}, inplace=True)
    result_df = result_df.merge(performance_df, how="inner", on="index")
    result_df.sort_values(by="index", inplace=True)
    result_df.reset_index(drop=True)
    result_df.to_csv(
        result_path,
        sep="\t",
        index=False,
        header=True,
    )
