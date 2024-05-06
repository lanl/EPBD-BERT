import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning
import lightning.pytorch.loggers
from sklearn import metrics

from epbd_bert.utility.data_utils import compute_multi_class_weights
from epbd_bert.utility.dnabert2 import get_dnabert2_pretrained_model
from epbd_bert.dnabert2_epbd.configs import Configs


class Dnabert2EPBDModel(lightning.LightningModule):
    """_summary_

    Args:
        lightning (_type_): _description_
    """

    def __init__(self, configs: Configs):
        """_summary_

        Args:
            configs (Configs): _description_
        """
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs

        self.model = get_dnabert2_pretrained_model()
        self.pooled_dropout = nn.Dropout(0.3)

        epbd_feature_proj_dim = int(configs.epbd_feature_input_dim / 2)
        self.epbd_feat_proj = nn.Sequential(
            nn.Linear(configs.epbd_feature_input_dim, epbd_feature_proj_dim),
            nn.Dropout(0.3),
            nn.ReLU(),
        )

        n_features = 768 + epbd_feature_proj_dim  # aggregated feature size
        self.classifier = nn.Linear(n_features, configs.n_classes)  #

        self.criterion = torch.nn.BCEWithLogitsLoss(
            weight=compute_multi_class_weights()
        )
        self.val_aucrocs = []
        self.val_losses = []

    def forward(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        labels = inputs.pop("labels")
        epbd_features = inputs.pop("epbd_features")

        # extracting dnabert2 features
        outputs = self.model(**inputs)
        # print(outputs)
        pooled_output = outputs[1]
        pooled_output = self.pooled_dropout(pooled_output)
        # print(pooled_output.shape)  # batch_size, 768

        # non-linear projection of epbd features
        epbd_features = self.epbd_feat_proj(epbd_features)

        # concatenating features
        features = torch.hstack([pooled_output, epbd_features])
        # print(features.shape)

        # applying attention
        # attn_w = self.attn_weights(features)
        # features = features * attn_w

        # # classification layer
        logits = self.classifier(features)
        # print(logits.shape, labels.shape)
        return logits, labels

    def calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """_summary_

        Args:
            logits (torch.Tensor): _description_
            targets (torch.Tensor): _description_

        Returns:
            float: _description_
        """
        loss = self.criterion(logits, targets)
        return loss

    def training_step(self, batch, batch_idx) -> float:
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            float: _description_
        """
        logits, targets = self.forward(batch)
        loss = self.calculate_loss(logits, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx) -> float:
        """_summary_

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            float: _description_
        """
        logits, targets = self.forward(batch)
        loss = self.calculate_loss(logits, targets)
        probs = F.sigmoid(logits)  # or softmax, depending on your problem and setup
        # print(labels[:10],probs[:10])
        auc_roc = metrics.roc_auc_score(
            targets.detach().cpu().numpy(),
            probs.detach().cpu().numpy(),
            average="micro",
        )

        self.val_aucrocs.append(auc_roc)
        self.val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        """_summary_"""
        val_avg_aucroc = torch.Tensor(self.val_aucrocs).mean()
        val_avg_loss = torch.Tensor(self.val_losses).mean()
        self.log_dict(
            dict(val_aucroc=val_avg_aucroc, val_loss=val_avg_loss),
            sync_dist=False,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.val_aucrocs.clear()
        self.val_losses.clear()

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=self.configs.weight_decay,
        )
