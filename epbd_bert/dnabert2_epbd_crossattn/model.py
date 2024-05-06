import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import lightning
import lightning.pytorch.loggers

from epbd_bert.utility.data_utils import compute_multi_class_weights
from epbd_bert.utility.dnabert2 import get_dnabert2_pretrained_model
from epbd_bert.dnabert2_epbd_crossattn.configs import EPBDConfigs


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MultiModalLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p_dropout=0.3):
        super(MultiModalLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=p_dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=p_dropout,
            batch_first=True,
        )
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.epbd_embedding_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, epbd_embedding, seq_embedding, key_padding_mask=None):
        # b: batch_size, l1: enc_batch_seq_len, l2: epbd_seq_len  d_model: embedding_dim
        # seq_embedding: b, l1, d_model
        # epbd_embedding: b, l2, d_model
        attn_output, self_attn_weights = self.self_attn(
            epbd_embedding, epbd_embedding, epbd_embedding
        )
        epbd_embedding = self.epbd_embedding_norm(
            epbd_embedding + self.dropout(attn_output)
        )

        # print(epbd_embedding.shape, seq_embedding.shape)
        attn_output, cross_attn_weights = self.cross_attn(
            query=epbd_embedding,
            key=seq_embedding,
            value=seq_embedding,
            key_padding_mask=key_padding_mask,
        )
        # print("cross-attn-out", attn_output)
        epbd_embedding = self.cross_attn_norm(
            epbd_embedding + self.dropout(attn_output)
        )

        ff_output = self.feed_forward(epbd_embedding)
        epbd_embedding = self.norm(epbd_embedding + self.dropout(ff_output))
        return epbd_embedding, self_attn_weights, cross_attn_weights


# batch_size, enc_batch_seq_len, epbd_seq_len = 4, 10, 8
# d_model, num_heads, d_ff, p_dropout = 16, 4, 32, 0.3
# m = MultiModalLayer(d_model, num_heads, d_ff, p_dropout)
# seq_embed = torch.rand(batch_size, enc_batch_seq_len, d_model)
# epbd_embed = torch.rand(batch_size, epbd_seq_len, d_model)
# key_padding_mask = torch.rand(batch_size, enc_batch_seq_len) < 0.5
# print(epbd_embed.shape, seq_embed.shape, key_padding_mask.shape)
# o = m(epbd_embed, seq_embed, key_padding_mask)
# print(o["outputs .shape)


class EPBDEmbedder(nn.Module):
    def __init__(self, in_channels, d_model, kernel_size=9):
        super(EPBDEmbedder, self).__init__()
        self.epbd_feature_extractor = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding="same",
        )

    def forward(self, x):
        # x is epbd_features: batch_size, in_channels, epbd_seq_len
        x = self.epbd_feature_extractor(x)  # batch_size, d_model, epbd_seq_len
        x = x.permute(0, 2, 1)  # batch_size, epbd_seq_len, d_model
        return x


# m = EPBDEmbedder(6, 9, 3)
# inp = torch.rand(10, 6, 15)
# o = m(inp)
# print(o.shape)


class PoolingLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.3) -> None:
        super(PoolingLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # batch_size, seq_len, embedding_dim
        x = torch.mean(x, dim=1)  # applying mean pooling
        x = self.fc(x)
        x = self.dropout(x)
        x = torch.relu(x)
        # print(x.shape)# batch_size, d_model
        return x


class EPBDDnabert2Model(lightning.LightningModule):
    def __init__(self, configs: EPBDConfigs):
        """_summary_

        Args:
            configs (EPBDConfigs): _description_
        """
        super(EPBDDnabert2Model, self).__init__()
        self.save_hyperparameters()

        self.seq_encoder = get_dnabert2_pretrained_model()
        self.epbd_embedder = EPBDEmbedder(
            in_channels=configs.epbd_feature_channels,
            d_model=configs.d_model,
            kernel_size=configs.epbd_embedder_kernel_size,
        )
        self.multi_modal_layer = MultiModalLayer(
            d_model=configs.d_model,
            num_heads=configs.num_heads,
            d_ff=configs.d_ff,
            p_dropout=configs.p_dropout,
        )
        self.pooling_layer = PoolingLayer(
            d_model=configs.d_model, dropout=configs.p_dropout
        )

        self.classifier = nn.Linear(configs.d_model, configs.n_classes)
        self.criterion = torch.nn.BCEWithLogitsLoss(
            weight=compute_multi_class_weights()
        )
        self.configs = configs

        self.val_aucrocs = []
        self.val_losses = []

    def forward(self, inputs):
        """_summary_

        Args:
            inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        targets = inputs.pop("labels")
        epbd_features = inputs.pop("epbd_features")

        seq_embedding = self.seq_encoder(**inputs)[0]
        # print("seq_embed:", seq_embedding.shape)  # b, batch_seq_len, d_model
        # raise

        attention_mask = inputs.pop("attention_mask")
        attention_mask = ~attention_mask.bool()
        epbd_embedding = self.epbd_embedder(epbd_features)
        # print("epbd_embed", epbd_embedding.shape)  # b, 200, d_model

        multi_modal_out, self_attn_weights, cross_attn_weights = self.multi_modal_layer(
            epbd_embedding,
            seq_embedding,
            attention_mask,
        )
        # print(
        #     f"multi_modal_out: {multi_modal_out.shape}",  # b, 200, d_model
        #     f"self_attn_weights: {self_attn_weights.shape}",  # b, 200, 200
        #     f"cross_attn_weights: {cross_attn_weights.shape}",  # b, 200, batch_seq_len
        # )

        pooled_output = self.pooling_layer(multi_modal_out)
        # print(pooled_output.shape)  # batch_size, 768
        logits = self.classifier(pooled_output)
        # print(logits.shape, targets.shape)
        return logits, targets

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
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=False)
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

    @classmethod
    def load_pretrained_model(self, checkpoint_path, mode="eval"):
        """_summary_

        Args:
            checkpoint_path (_type_): _description_
            mode (str, optional): _description_. Defaults to "eval".

        Returns:
            _type_: _description_
        """
        # mode: eval, train
        model = self.load_from_checkpoint(checkpoint_path)
        # model.to(device)
        if mode == "eval":
            model.eval()
        return model


# Q: batch_size, seq_len1, d_model
# K, V: batch_size, seq_len2, d_model=786
# batch_size, seq_len2, 6 # coord+flips
