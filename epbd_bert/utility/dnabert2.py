from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from epbd_bert.path_configs import dnabert2_pretrained_dirpath


def get_dnabert2_tokenizer(max_num_tokens=512, home_dir=""):
    tokenizer = AutoTokenizer.from_pretrained(
        home_dir + dnabert2_pretrained_dirpath,
        model_max_length=max_num_tokens,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    return tokenizer


def get_dnabert2_pretrained_model(home_dir=""):
    model = AutoModel.from_pretrained(home_dir + dnabert2_pretrained_dirpath, trust_remote_code=True)
    return model


def load_dnabert2_for_classification(num_labels, home_dir=""):
    model = AutoModelForSequenceClassification.from_pretrained(
        home_dir + dnabert2_pretrained_dirpath,
        num_labels=num_labels,
        trust_remote_code=True,
    )
    return model
