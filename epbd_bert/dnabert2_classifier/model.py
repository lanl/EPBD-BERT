import torch
import torch.nn.functional as F
import lightning
import lightning.pytorch.loggers
from sklearn import metrics

from epbd_bert.utility.data_utils import compute_multi_class_weights
from epbd_bert.utility.dnabert2 import load_dnabert2_for_classification
from epbd_bert.dnabert2_classifier.configs import Configs


# DNABERT2 tf-dna binding classifier
class DNABERT2Classifier(lightning.LightningModule):
    def __init__(self, configs: Configs):
        """_summary_

        Args:
            configs (Configs): _description_
        """
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        # self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_dnabert2_for_classification(num_labels=configs.n_classes)

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
        # forward pass
        # print(inputs["input_ids"].shape, inputs["attention_mask"].shape, labels.shape)
        outputs = self.model(**inputs)
        # print(outputs)
        logits = outputs.get("logits")
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
        self.log(
            "train_loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=False
        )
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
        """

        Returns:
            _type_: _description_
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.configs.learning_rate,
            weight_decay=self.configs.weight_decay,
        )
