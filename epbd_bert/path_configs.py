# data_home_dir = "epbd-bert/"  # "./"
data_home_dir = ""
dnabert2_pretrained_dirpath = data_home_dir + "resources/DNABERT-2-117M/"
# print(dnabert2_pretrained_dirpath)

# pydnaepbd_features_path = "data/pydnaepbd_things/coord_flips/id_seqs/"
# pydnaepbd_features_path = "gen-epbd/cond_epbd/coord_flips/"

train_data_filepath = data_home_dir + "resources/train_val_test/peaks_with_labels_train.tsv.gz"
val_data_filepath = data_home_dir + "resources/train_val_test/peaks_with_labels_val.tsv.gz"
test_data_filepath = data_home_dir + "resources/train_val_test/peaks_with_labels_test.tsv.gz"

dnabert2_classifier_ckptpath = data_home_dir + "resources/trained_weights/dnabert2_classifier.ckpt"
epbd_dnabert2_ckptpath = data_home_dir + "resources/trained_weights/epbd_dnabert2.ckpt"
epbd_dnabert2_crossattn_ckptpath = data_home_dir + "resources/trained_weights/epbd_dnabert2_crossattn.ckpt"
epbd_dnabert2_crossattn_best_ckptpath = data_home_dir + "resources/trained_weights/epbd_dnabert2_crossattn_best.ckpt"
