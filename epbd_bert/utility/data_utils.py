import torch
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import epbd_bert.utility.pickle_utils as pickle_utils


# reading and processing peakfiles metadata to generate the labels dict
def get_uniform_peaks_metadata(home_dir="/usr/projects/pyDNA_EPBD/tf_dna_binding/"):
    col_names = [
        "filename",
        "project",
        "lab",
        "composite",
        "dataType",
        "view",
        "cell",
        "treatment",
        "antibody",
        "control",
        "dataVersion",
        "dccAccession",
        "controlId",
        "quality",
        "tableName",
        "type",
        "md5sum",
        "size",
    ]
    with open(home_dir + "data/downloads/wgEncodeAwgTfbsUniform/files.txt", "r") as h:
        data = []
        for line in h.readlines():
            # print(line.strip().split(";"))
            line_items = line.strip().split(";")

            row = {}
            filename, project = line_items[0].split("\t")
            project = project.split("=")[1]
            # print(filename, project)

            row["filename"] = filename
            row["project"] = project

            for line_item in line_items[1:]:
                key, value = line_item.strip().split("=")
                row[key] = value

            data.append(row)
            # print(row)
            # break
    peaks_metadata_df = pd.DataFrame.from_records(data)
    print(peaks_metadata_df.shape)
    print(peaks_metadata_df.columns)
    return peaks_metadata_df


def compute_multi_class_weights(home_dir=""):
    data_path = home_dir + "resources/train_val_test/peaks_with_labels_train.tsv.gz"
    data_df = pd.read_csv(data_path, compression="gzip", sep="\t")
    labels_dict = pickle_utils.load(home_dir + "resources/processed_data/peakfilename_index_dict.pkl")

    all_labels = []

    def get_all_labels(labels):
        for l in labels.split(","):
            l = l.strip()
            all_labels.append(labels_dict[l])

    data_df["labels"].apply(get_all_labels)
    class_weights = compute_class_weight("balanced", classes=np.array(list(range(len(labels_dict)))), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # print(class_weights.shape)
    return class_weights


# compute_multi_class_weights()

def compute_binary_class_weights(data_index=0):
    data_df = pd.read_csv(f"../data/mouse_tfbs/mouse/{data_index}/train.csv")
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=data_df["label"])
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    # print(class_weights)
    return class_weights

# compute_binary_class_weights(4)