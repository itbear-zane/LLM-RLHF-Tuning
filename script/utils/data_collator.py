# 数据整理器模块 - 负责将原始数据批处理成模型输入格式
#
# 本模块提供了多种数据整理器：
# 1. DataCollatorForSupervisedDataset: 用于监督学习的数据整理
# 2. PPODataCollatorWithPadding: 用于PPO训练的数据整理，支持倒序prompt
# 3. PairDataCollatorWithPadding: 用于成对比较的数据整理

import torch
import transformers
from transformers import (
    DataCollatorWithPadding, BatchEncoding, PreTrainedTokenizerBase,
    PreTrainedModel, PreTrainedTokenizer
)
from typing import Optional, Dict, Sequence, Union, List, Any
from dataclasses import dataclass

# 损失计算中忽略的标签索引
IGNORE_INDEX=-100

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.LongTensor(instance[key]) for instance in instances] for key in ("input_ids", "label_ids"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).long(),
        )

class PairDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, instances: Sequence[Dict[str, Union[torch.Tensor, Sequence[int]]]]) -> Dict[str, torch.Tensor]:

        accepts_ids, accepts_labels, rejects_ids, rejects_labels = [], [], [], []
        for instance in instances:
            length = len(instance["input_ids"]) // 2 
            accepts_id = instance["input_ids"][:length]
            rejects_id = instance["input_ids"][length:]
            accepts_label = instance["label_ids"][:length]
            rejects_label = instance["label_ids"][length:]

            accepts_ids.append(torch.LongTensor(accepts_id))
            accepts_labels.append(torch.LongTensor(accepts_label))
            rejects_ids.append(torch.LongTensor(rejects_id))
            rejects_labels.append(torch.LongTensor(rejects_label))
            
        accepts_input_ids = torch.nn.utils.rnn.pad_sequence(
            accepts_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        accepts_labels = torch.nn.utils.rnn.pad_sequence(accepts_labels, batch_first=True, padding_value=IGNORE_INDEX)
        rejects_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejects_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        rejects_labels = torch.nn.utils.rnn.pad_sequence(rejects_labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            accepts_input_ids=accepts_input_ids, 
            accepts_labels=accepts_labels, 
            accepts_attention_mask=accepts_input_ids.ne(self.tokenizer.pad_token_id).long(),
            rejects_input_ids=rejects_input_ids,
            rejects_labels=rejects_labels,
            rejects_attention_mask=rejects_input_ids.ne(self.tokenizer.pad_token_id).long(),
        )



class PPODataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:

        input_ids = [torch.LongTensor(instance["input_ids"]).flip(0) for instance in instances] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        return dict(
            input_ids=input_ids.flip(1),   
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).flip(1).long(),
        )   # prompt 倒序 

        




