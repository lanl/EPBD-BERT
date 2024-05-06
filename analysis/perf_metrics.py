import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import utility.pickle_utils as pickle_utils


def get_auroc(preds, obs):
    fpr, tpr, ths = metrics.roc_curve(obs, preds, drop_intermediate=False)
    auroc = metrics.auc(fpr, tpr)
    return auroc, fpr, tpr, ths


def get_aupr(preds, obs):
    precision, recall, ths = metrics.precision_recall_curve(obs, preds)
    aupr = metrics.auc(recall, precision)
    return aupr, precision, recall, ths


def get_aurocs_and_auprs(pred_probs: np.array, targets: np.array, verbose=False,fpath='/usr/projects/pyDNA_EPBD/tf_dna_binding/'):
    label2index_dict = pickle_utils.load(fpath+"data/processed/peakfilename_index_dict.pkl")
    index2label_dict = {i: label for label, i in label2index_dict.items()}
    # index2label_dict[0]
    auroc_dict = {}
    aupr_dict = {}
    for task_id in range(pred_probs.shape[1]):
        pred, obs = pred_probs[:, task_id], targets[:, task_id]

        auroc, _, _, _ = get_auroc(pred, obs)
        auroc = round(auroc, 5)
        aupr, _, _, _ = get_aupr(pred, obs)
        aupr = round(aupr, 5)

        auroc_dict[index2label_dict[task_id]] = auroc
        aupr_dict[index2label_dict[task_id]] = aupr
    return auroc_dict, aupr_dict


def compute_predictions(model, dl: DataLoader, preds_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(
        preds_and_targets_dict["preds"].shape,
        preds_and_targets_dict["targets"].shape,
    )
    pickle_utils.save(preds_and_targets_dict, preds_path)
    return preds_and_targets_dict


def get_predictions(model_name: str, data_type: str, compute_again=False, home_dir=""):
    preds_dir = home_dir + f"analysis/preds/"
    os.makedirs(preds_dir, exist_ok=True)
    preds_path = preds_dir + f"{model_name}_on_{data_type}.pkl"

    if os.path.exists(preds_path) and not compute_again:
        preds_and_targets_dict = pickle_utils.load(preds_path)
        print(
            preds_and_targets_dict["preds"].shape,
            preds_and_targets_dict["targets"].shape,
        )
        return preds_and_targets_dict
    else:
        from utility.dnabert2 import get_dnabert2_tokenizer
        from analysis.models_factory import get_model_and_dataloader

        data_path = (
            home_dir + f"data/train_val_test/peaks_with_labels_{data_type}.tsv.gz"
        )
        tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
        model, dl = get_model_and_dataloader(model_name, data_path, tokenizer)
        return compute_predictions(model, dl, preds_path)
