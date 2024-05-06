import torch
from torch.utils.data import DataLoader
import transformers
from datasets.data_collators import SeqLabelEPBDDataCollator, SeqLabelDataCollator


def get_model_and_dataloader(
    model_name,
    data_path,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_size=64,
    num_workers=8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "dnabert2":
        from dnabert2_classifier.model import DNABERT2Classifier
        from datasets.sequence_dataset import SequenceDataset

        ckpt_path = "dnabert2_classifier/backups/version_2/checkpoints/epoch=20-step=50547-val_loss=0.052-val_aucroc=0.939.ckpt"
        model = DNABERT2Classifier.load_from_checkpoint(ckpt_path)

        ds = SequenceDataset(data_path, tokenizer)
        data_collator = SeqLabelDataCollator(pad_token_id=tokenizer.pad_token_id)

    elif model_name == "dnabert2_epbd":
        from dnabert2_epbd.model import Dnabert2EPBDModel
        from datasets.sequence_epbd_dataset import SequenceEPBDDataset

        ckpt_path = "dnabert2_epbd/backups/version_0/checkpoints/epoch=17-step=43326-val_loss=0.053-val_aucroc=0.938.ckpt"
        model = Dnabert2EPBDModel.load_from_checkpoint(ckpt_path)

        ds = SequenceEPBDDataset(data_path, tokenizer)
        data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)

    elif model_name == "dnabert2_randepbd":
        from dnabert2_epbd.model import Dnabert2EPBDModel
        from datasets.sequence_randepbd_dataset import SequenceRandEPBDDataset

        ckpt_path = "dnabert2_epbd/backups/version_1/checkpoints/epoch=18-step=45733-val_loss=0.053-val_aucroc=0.938.ckpt"
        model = Dnabert2EPBDModel.load_from_checkpoint(ckpt_path)

        ds = SequenceRandEPBDDataset(data_path, tokenizer)
        data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)

    elif model_name == "dnabert2_epbd_crossattn_bestloss":
        from dnabert2_epbd_crossattn.model import EPBDDnabert2Model
        from datasets.sequence_epbd_multimodal_dataset import (
            SequenceEPBDMultiModalDataset,
        )

        ckpt_path = (
            "dnabert2_epbd_crossattn/backups/v1/checkpoints/epoch=12-step=125151.ckpt"
        )
        model = EPBDDnabert2Model.load_from_checkpoint(ckpt_path)

        ds = SequenceEPBDMultiModalDataset(data_path, tokenizer)
        data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)

    elif model_name == "dnabert2_epbd_crossattn_bestauroc":
        from dnabert2_epbd_crossattn.model import EPBDDnabert2Model
        from datasets.sequence_epbd_multimodal_dataset import (
            SequenceEPBDMultiModalDataset,
        )

        ckpt_path = "dnabert2_epbd_crossattn/backups/v2/checkpoints/epoch=29-step=72210-val_loss=0.058-val_auc_roc=0.939.ckpt"
        model = EPBDDnabert2Model.load_from_checkpoint(ckpt_path)

        ds = SequenceEPBDMultiModalDataset(data_path, tokenizer)
        data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)

    elif model_name == "dnabert2_randepbd_crossattn":
        from dnabert2_epbd_crossattn.model import EPBDDnabert2Model
        from datasets.sequence_randepbd_multimodal_dataset import (
            SequenceRandEPBDMultiModalDataset,
        )

        ckpt_path = "dnabert2_epbd_crossattn/backups/v3/checkpoints/epoch=14-step=36105-val_loss=0.056-val_aucroc=0.935.ckpt"
        model = EPBDDnabert2Model.load_from_checkpoint(ckpt_path)

        ds = SequenceRandEPBDMultiModalDataset(data_path, tokenizer)
        data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)
    elif model_name == "dnabert2_epbd_crossattn_bestmodel":
        from dnabert2_epbd_crossattn.model import EPBDDnabert2Model
        from datasets.sequence_epbd_multimodal_dataset import (
            SequenceEPBDMultiModalDataset,
        )

        ckpt_path = "analysis/best_model/epoch=9-step=255700.ckpt"
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
# from utility.dnabert2 import get_dnabert2_tokenizer

# tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model, dl = get_model_and_dataloader(
#     model_name="dnabert2_epbd_crossattn_bestmodel",
#     data_path="data/train_val_test/peaks_with_labels_test.tsv.gz",
#     tokenizer=tokenizer,
# )

# for i, batch in enumerate(dl):
#     x = {key: batch[key].to(device) for key in batch.keys()}
#     logits, targets = model(x)
#     print(i, logits.shape, targets.shape)
#     break
