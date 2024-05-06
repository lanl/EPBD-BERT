from dataclasses import dataclass
import torch

torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()


@dataclass(kw_only=True)
class TrainingConfigs:
    batch_size: int = 32  # 170,
    num_workers: int = 1  # 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_epochs: int = 5


@dataclass(kw_only=True)
class EPBDConfigs(TrainingConfigs):
    d_model: int = 256
    epbd_feature_channels: int = 1  # only coordinates
    epbd_embedder_kernel_size: int = 11
    num_heads: int = 8
    d_ff: int = 768
    p_dropout: float = 0.1
    need_weights: bool = False
    n_classes: int = 690
    best_model_monitor: str = "val_loss"
    best_model_monitor_mode: str = "min"
