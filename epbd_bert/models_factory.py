import torch
from torch.utils.data import DataLoader
import transformers
from epbd_bert.datasets.data_collators import (
    SeqLabelEPBDDataCollator,
    SeqLabelDataCollator,
)

from epbd_bert.path_configs import (
    dnabert2_classifier_ckptpath,
    epbd_dnabert2_ckptpath,
    epbd_dnabert2_crossattn_ckptpath,
    epbd_dnabert2_crossattn_best_ckptpath,
)


def get_model_and_dataloader(
    model_name,
    data_path,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size=64,
    num_workers=8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "dnabert2_classifier":
        from epbd_bert.dnabert2_classifier.model import DNABERT2Classifier
        from epbd_bert.datasets.sequence_dataset import SequenceDataset

        # ckpt_path = "dnabert2_classifier/backups/version_2/checkpoints/epoch=20-step=50547-val_loss=0.052-val_aucroc=0.939.ckpt"
        ckpt_path = dnabert2_classifier_ckptpath
        model = DNABERT2Classifier.load_from_checkpoint(ckpt_path)

        ds = SequenceDataset(data_path, tokenizer)
        data_collator = SeqLabelDataCollator(pad_token_id=tokenizer.pad_token_id)

    elif model_name == "epbd_dnabert2":
        from epbd_bert.dnabert2_epbd.model import Dnabert2EPBDModel
        from epbd_bert.datasets.sequence_epbd_dataset import SequenceEPBDDataset

        # ckpt_path = "dnabert2_epbd/backups/version_0/checkpoints/epoch=17-step=43326-val_loss=0.053-val_aucroc=0.938.ckpt"
        ckpt_path = epbd_dnabert2_ckptpath
        model = Dnabert2EPBDModel.load_from_checkpoint(ckpt_path)

        ds = SequenceEPBDDataset(data_path, tokenizer)
        data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)

    elif model_name == "epbd_dnabert2_crossattn_best":
        from epbd_bert.dnabert2_epbd_crossattn.model import EPBDDnabert2Model
        from epbd_bert.datasets.sequence_epbd_multimodal_dataset import (
            SequenceEPBDMultiModalDataset,
        )

        # ckpt_path = "analysis/best_model/epoch=9-step=255700.ckpt" # best model
        ckpt_path = epbd_dnabert2_crossattn_best_ckptpath
        model = EPBDDnabert2Model.load_from_checkpoint(ckpt_path)

        ds = SequenceEPBDMultiModalDataset(data_path, tokenizer)
        data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)

    model.to(device)
    model.eval()
    dl = DataLoader(
        ds,
        collate_fn=data_collator,
        shuffle=False,
        pin_memory=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print("DS, DL:", ds.__len__(), len(dl))
    return model, dl


# test all models with corresponding test dataloader
# from epbd_bert.utility.dnabert2 import get_dnabert2_tokenizer
# from epbd_bert.path_configs import test_data_filepath

# tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model, dl = get_model_and_dataloader(
#     model_name="epbd_dnabert2_crossattn_best",
#     data_path=test_data_filepath,
#     tokenizer=tokenizer,
# )

# for i, batch in enumerate(dl):
#     x = {key: batch[key].to(device) for key in batch.keys()}
#     logits, targets = model(x)
#     print(i, logits.shape, targets.shape)
#     break


# checkpoint = torch.load(
#     dnabert2_classifier_ckptpath, map_location=lambda storage, loc: storage
# )
# print(checkpoint.keys())
# print(checkpoint["hyper_parameters"])
# print(checkpoint["state_dict"])
# {"learning_rate": the_value, "another_parameter": the_other_value}
