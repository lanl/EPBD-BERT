import torch
from dnabert2_epbd_crossattn.model import EPBDDnabert2Model
from dnabert2_epbd_crossattn.configs import EPBDConfigs


class EPBDDnabert2ModelForAnalysis(EPBDDnabert2Model):
    def __init__(self, configs: EPBDConfigs):
        super().__init__(configs)

    @torch.no_grad()
    def analysis_forward(self, inputs):
        """
        b, l1, l2, c = batch_size, enc_seq_len, epbd_seq_len, n_classes
        keys of inputs: input_ids, epbd_features, labels, attention_mask
            input_ids: [b, l1]
            attention_mask: [b, l1]
            epbd_features: [b, l2]
            labels: [b, c]
        Args:
            inputs (_type_): _description_
        """
        targets = inputs.pop("labels")
        epbd_features = inputs.pop("epbd_features")

        seq_embedding, pooled_embedding = self.seq_encoder(**inputs)
        # print("seq_embed:", seq_embedding.shape)  # b, l2, d_model
        # raise

        attention_mask = inputs.pop("attention_mask")
        attention_mask = ~attention_mask.bool()
        epbd_embedding = self.epbd_embedder(epbd_features)
        # print("epbd_embed", epbd_embedding.shape)  # b, 200, d_model
        avg_epbd_features = epbd_embedding.mean(dim=2).squeeze(0)
        # print(avg_epbd_features.shape)
        # raise

        multi_modal_out, self_attn_weights, cross_attn_weights = self.multi_modal_layer(
            epbd_embedding,
            seq_embedding,
            attention_mask,
        )
        # print(
        #     f"multi_modal_out: {multi_modal_out.shape}",  # b, 200, d_model
        #     f"self_attn_weights: {self_attn_weights.shape}",  # b, 200, 200
        #     f"cross_attn_weights: {cross_attn_weights.shape}",  # b, 200, l2
        # )
        # pickle_utils.save(
        #     self_attn_weights.detach().cpu().numpy(),
        #     "analysis/temp/self_attn_weights.pkl",
        # )
        # pickle_utils.save(
        #     cross_attn_weights.detach().cpu().numpy(),
        #     "analysis/temp/cross_attn_weights.pkl",
        # )

        pooled_output = self.pooling_layer(multi_modal_out)
        # print("pooled_out: ", pooled_output.shape)  # b, d_model
        logits = self.classifier(pooled_output)
        # print(logits.shape, targets.shape)
        return dict(
            logits=logits.detach().cpu().numpy(),
            targets=targets.detach().cpu().numpy(),
            self_attn_weights=self_attn_weights.squeeze(0).detach().cpu().numpy(),
            cross_attn_weights=cross_attn_weights.squeeze(0).detach().cpu().numpy(),
            avg_epbd_features=avg_epbd_features.detach().cpu().numpy(),
        )


# saving the conv kernel weights
# import utility.pickle_utils as pickle_utils

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # ckpt_path = "dnabert2_epbd_crossattn/backups/v1/checkpoints/epoch=12-step=125151.ckpt"
# ckpt_path = "analysis/best_model/epoch=9-step=255700.ckpt"
# model = EPBDDnabert2ModelForAnalysis.load_from_checkpoint(ckpt_path)
# # model.to(device)
# # model.eval()
# print(
#     model.epbd_embedder.epbd_feature_extractor.weight.shape,  # 768, feat_channels=6, kernel_size=7
#     model.epbd_embedder.epbd_feature_extractor.bias.shape,  # 768
# )
# pickle_utils.save(
#     model.epbd_embedder.epbd_feature_extractor.weight.detach().cpu().numpy(),
#     "analysis/weights/epbd_feature_extractor_kernel_weights.pkl",
# )
