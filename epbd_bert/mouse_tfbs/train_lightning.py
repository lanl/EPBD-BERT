from torch.utils.data import DataLoader

from ..dnabert2_epbd_crossattn.configs import EPBDConfigs
from ..dnabert2_epbd_crossattn.model import EPBDDnabert2Model
from ..datasets.data_collators import SeqLabelEPBDDataCollator
from ..utility.dnabert2 import get_dnabert2_tokenizer
from .mouse_sequence_epbd_dataset import MouseSequenceEPBDDataset

import lightning
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    configs = EPBDConfigs(
        batch_size=170,  # 170,
        num_workers=32,  # 32
        learning_rate=1e-5,
        weight_decay=0.1,
        max_epochs=50,  # 100
        d_model=768,
        epbd_feature_channels=6,  # coord+flips
        epbd_embedder_kernel_size=11,
        num_heads=8,
        d_ff=768,
        p_dropout=0.1,
        need_weights=False,
        n_classes=1,
        best_model_monitor="val_loss",
        best_model_monitor_mode="min",
    )

    data_index = 4 # 0, 1, 2, 3, 4
    tokenizer = get_dnabert2_tokenizer(max_num_tokens=512)
    data_collator = SeqLabelEPBDDataCollator(tokenizer.pad_token_id)
    train_dataset = MouseSequenceEPBDDataset(index=data_index, data_type="train", tokenizer=tokenizer)
    val_dataset = MouseSequenceEPBDDataset(index=data_index, data_type="dev", tokenizer=tokenizer)
    print("train|val DS sizes:", train_dataset.__len__(), val_dataset.__len__())
    train_dl = DataLoader(
        train_dataset, collate_fn=data_collator, shuffle=True, pin_memory=False, batch_size=configs.batch_size, num_workers=configs.num_workers
    )
    val_dl = DataLoader(
        val_dataset, collate_fn=data_collator, shuffle=False, pin_memory=False, batch_size=configs.batch_size, num_workers=configs.num_workers
    )
    print("train|val DL sizes:", len(train_dl), len(val_dl))

    model = EPBDDnabert2Model(configs)

    out_dir = f"/lustre/scratch4/turquoise/akabir/mouse_tfbs/{data_index}"
    csv_logger = CSVLogger(save_dir=out_dir)
    strategy = DDPStrategy(find_unused_parameters=True)
    checkpoint_callback = ModelCheckpoint(
        monitor=configs.best_model_monitor,
        mode=configs.best_model_monitor_mode,
        every_n_epochs=1,
        filename="{epoch}-{step}-{val_loss:.3f}-{val_aucroc:.3f}",
        save_last=True,
    )

    trainer = lightning.Trainer(
        devices="auto",  # 1, "auto"
        strategy=strategy,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=0.2,
        max_epochs=configs.max_epochs,  # 100,
        # limit_train_batches=5,  # cmnt out when full run
        # limit_val_batches=3,  # cmnt out when full run
        # val_check_interval=2000, # cmnt out when full run
        check_val_every_n_epoch=1,
        log_every_n_steps=20,  # 50,
        default_root_dir=out_dir,
        logger=csv_logger,
        callbacks=[checkpoint_callback],
    )

    print(trainer.num_devices, trainer.device_ids, trainer.strategy)
    trainer.fit(model, train_dl, val_dl)


# from tf_dna_binding/epbd-bert
# conda activate /usr/projects/pyDNA_EPBD/tf_dna_binding/.venvs/python311_conda_3
# python -m epbd_bert.mouse_tfbs.train_lightning
