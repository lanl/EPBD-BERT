import os
import torch
from torch.utils.data import DataLoader

from epbd_bert.datasets.sequence_epbd_dataset import SequenceEPBDDataset
from epbd_bert.datasets.data_collators import SeqLabelEPBDDataCollator

from epbd_bert.dnabert2_epbd.model import Dnabert2EPBDModel
from epbd_bert.dnabert2_epbd.configs import Configs
from epbd_bert.utility.dnabert2 import get_dnabert2_tokenizer

import lightning
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

# epbd_features_type = "real"
# if "epbd_features_type" in os.environ:
#     epbd_features_type = os.environ.get("epbd_features_type")
# configs = Configs(epbd_features_type=epbd_features_type)
if __name__ == "__main__":
    configs = Configs()

    tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
    data_collator = SeqLabelEPBDDataCollator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = SequenceEPBDDataset(
        data_path="resources/train_val_test/peaks_with_labels_train.tsv.gz",
        pydnaepbd_features_path="resources/pydnaepbd_things/coord_flips/id_seqs/",  # ../data, resources
        tokenizer=tokenizer,
    )
    val_dataset = SequenceEPBDDataset(
        data_path="resources/train_val_test/peaks_with_labels_val.tsv.gz",
        pydnaepbd_features_path="resources/pydnaepbd_things/coord_flips/id_seqs/",  # ../data, resources
        tokenizer=tokenizer,
    )
    print(train_dataset.__len__(), val_dataset.__len__())
    train_dl = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        shuffle=True,
        pin_memory=False,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
    )
    val_dl = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        shuffle=False,
        pin_memory=False,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
    )
    print(len(train_dl), len(val_dl))

    model = Dnabert2EPBDModel(configs)

    out_dir = "dnabert2_epbd/"
    csv_logger = CSVLogger(save_dir=out_dir)
    # strategy = DDPStrategy(find_unused_parameters=True)
    checkpoint_callback = ModelCheckpoint(
        monitor=configs.best_model_monitor,
        mode=configs.best_model_monitor_mode,
        every_n_epochs=1,
        filename="{epoch}-{step}-{val_loss:.3f}-{val_aucroc:.3f}",
        save_last=True,
        # auto_insert_metric_name=True,
    )
    trainer = lightning.Trainer(
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision="16-mixed",
        gradient_clip_val=0.2,
        max_epochs=configs.max_epochs,  # 100,
        # limit_train_batches=10,
        # limit_val_batches=10,
        # val_check_interval=2000,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,  # 50,
        default_root_dir=out_dir,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )

    print(trainer.num_devices, trainer.device_ids, trainer.strategy)
    trainer.fit(model, train_dl, val_dl)
