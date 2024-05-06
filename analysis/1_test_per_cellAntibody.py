import os
import numpy as np
import pandas as pd
from sklearn import metrics

from analysis.perf_metrics import (
    get_auroc,
    get_aupr,
    get_aurocs_and_auprs,
    get_predictions,
)

from utility.data_utils import get_uniform_peaks_metadata

if __name__ == "__main__":
    # change these 2 lines only
    model_name = "dnabert2_epbd_crossattn_bestmodel"
    data_type = "test"

    preds_and_targets_dict = get_predictions(model_name, data_type, compute_again=False)

    # -----overall auc-roc
    auc_roc = metrics.roc_auc_score(
        preds_and_targets_dict["targets"],
        preds_and_targets_dict["preds"],
        average="micro",
    )
    print("overall auc-roc: ", auc_roc)

    # -----computing auc-roc and auc-pr per cell/antibody type----
    auroc_dict, aupr_dict = get_aurocs_and_auprs(
        preds_and_targets_dict["preds"],
        preds_and_targets_dict["targets"],
        True,
    )
    auroc_list = [v for k, v in auroc_dict.items()]
    aupr_list = [v for k, v in aupr_dict.items()]
    print("Averaged AUROC:", np.nanmean(auroc_list))
    print("Averaged AUPR:", np.nanmean(aupr_list))

    # -------computing AUROC and AUPR per cell-antibody
    # post-processing computed auroc and aupr
    def get_df_from_dict(a_dict, value_col_name):
        df = pd.DataFrame.from_dict(a_dict, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={0: value_col_name}, inplace=True)
        return df

    auroc_df = get_df_from_dict(auroc_dict, value_col_name=f"{model_name}_auroc")
    aupr_df = get_df_from_dict(aupr_dict, value_col_name=f"{model_name}_aupr")
    performance_df = pd.merge(left=auroc_df, right=aupr_df, how="inner", on="index")
    # performance_df

    # loading labels-metadata which contains cell and antibody type
    peaks_metadata_df = get_uniform_peaks_metadata()
    result_df = peaks_metadata_df[["tableName", "cell", "antibody"]].copy()
    result_df.rename(columns={"tableName": "index"}, inplace=True)

    # merging performance and labels metadata
    result_df = result_df.merge(performance_df, how="inner", on="index")
    result_df.sort_values(by="index", inplace=True)
    result_df.reset_index(drop=True)

    # saving the result
    results_dir = "analysis/aurocs_auprs/"
    os.makedirs(results_dir, exist_ok=True)
    result_df.to_csv(
        results_dir + f"{model_name}_on_{data_type}.tsv",
        sep="\t",
        index=False,
        header=True,
    )
