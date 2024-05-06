from dataclasses import dataclass
import torch

torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()


@dataclass(kw_only=True)
class Configs:
    n_classes: int = 690
    batch_size: int = 170  # 32, 170,
    num_workers: int = 32  # 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    max_epochs: int = 100  # 5, 100
    best_model_monitor: str = "val_loss"
    best_model_monitor_mode: str = "min"
