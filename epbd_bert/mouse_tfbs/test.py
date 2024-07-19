from ..dnabert2_epbd_crossattn.model import EPBDDnabert2Model
from ..datasets.data_collators import SeqLabelEPBDDataCollator
from .mouse_sequence_epbd_dataset import MouseSequenceEPBDDataset
from ..utility.dnabert2 import get_dnabert2_tokenizer

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef

device = "cuda" if torch.cuda.is_available() else "cpu"

data_index = 4 # 0, 1, 2, 3, 4
tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
data_collator = SeqLabelEPBDDataCollator(tokenizer.pad_token_id)
test_dataset = MouseSequenceEPBDDataset(index=data_index, data_type="test", tokenizer=tokenizer)
test_dl = DataLoader(test_dataset, collate_fn=data_collator, shuffle=False, pin_memory=False, batch_size=64, num_workers=10)
print("test DS|DL size:", test_dataset.__len__(), len(test_dl))

# 0: epoch=3-step=156-val_loss=0.429-val_aucroc=0.895.ckpt
# 1: epoch=4-step=400-val_loss=0.169-val_aucroc=0.983.ckpt
# 2: epoch=18-step=76-val_loss=0.319-val_aucroc=0.957.ckpt
# 3: epoch=28-step=87-val_loss=0.361-val_aucroc=0.944.ckpt
# 4: epoch=7-step=184-val_loss=0.514-val_aucroc=0.841.ckpt

checkpoint_name = "epoch=7-step=184-val_loss=0.514-val_aucroc=0.841.ckpt"
checkpoint_path = f"/lustre/scratch4/turquoise/akabir/mouse_tfbs/{data_index}/lightning_logs/version_0/checkpoints/{checkpoint_name}"
model = EPBDDnabert2Model.load_pretrained_model(checkpoint_path, mode="eval")


all_preds, all_targets = [], []
for i, batch in enumerate(test_dl):
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
all_preds, all_targets = np.vstack(all_preds).squeeze(1), np.vstack(all_targets).squeeze(1)
print(all_preds.shape, all_targets.shape)
all_preds = np.where(all_preds > 0.5, 1, 0)

print(matthews_corrcoef(all_targets, all_preds))


# from tf_dna_binding/epbd-bert
# conda activate /usr/projects/pyDNA_EPBD/tf_dna_binding/.venvs/python311_conda_3
# python -m epbd_bert.mouse_tfbs.test

# .5828, .8527, .8054, .7013, .48950
# 0.5828599831997519
# 0.8527706874885341
# 0.8054172503325737
# 0.7013031178127139
# 0.48954405219972724
