from typing import Dict, Sequence
from dataclasses import dataclass
import torch


@dataclass
class SeqLabelEPBDDataCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, epbd_features = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "epbd_features")
        )

        # padding tokens in a mini-batch as the length of the maximum seq_len
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        # print(input_ids.shape)

        epbd_features = torch.stack(epbd_features)

        # stacking labels
        labels = torch.stack(labels)
        labels = labels.to(dtype=torch.float32)

        # setting up attention mask
        attention_mask = input_ids.ne(self.pad_token_id).int()
        return dict(
            input_ids=input_ids,
            epbd_features=epbd_features,
            labels=labels,
            attention_mask=attention_mask,
        )


# dc = SeqLabelEPBDDataCollator(pad_token_id=100)
# x = [
#     dict(
#         input_ids=torch.ones(10), labels=torch.ones(3), epbd_features=torch.rand(1200)
#     ),
#     dict(input_ids=torch.ones(7), labels=torch.ones(3), epbd_features=torch.rand(1200)),
#     dict(input_ids=torch.ones(3), labels=torch.ones(3), epbd_features=torch.rand(1200)),
# ]
# print(dc(x))


@dataclass
class SeqLabelDataCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        # padding tokens in a mini-batch as the length of the maximum seq_len
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        # print(input_ids.shape)

        # stacking labels
        labels = torch.stack(labels)
        labels = labels.to(dtype=torch.float32)

        # setting up attention mask
        attention_mask = input_ids.ne(self.pad_token_id).int()
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


# dc = SeqLabelDataCollator()
# x = [
#     dict(input_ids=torch.ones(10), labels=torch.ones(3)),
#     dict(input_ids=torch.ones(7), labels=torch.ones(3)),
#     dict(input_ids=torch.ones(3), labels=torch.ones(3)),
# ]
# print(dc(x))
