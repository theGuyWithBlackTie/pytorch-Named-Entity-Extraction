import config
import torch
import transformers
import torch.nn as nn

class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert    = transformers.BertModel.from_pretrained(config.BERT_MODEL_PATH, return_dict=False)
        self.dropout1= nn.Dropout(0.3)
        self.dropout2= nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)

    def forward(self, ids, mask, token_type_ids, target_pos, target_tag):
        seq_output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        tag_dropout = self.dropout1(seq_output)
        pos_dropout = self.dropout2(seq_output)

        tag = self.out_tag(tag_dropout)
        pos = self.out_pos(pos_dropout)

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return tag, pos, loss


def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss   = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss