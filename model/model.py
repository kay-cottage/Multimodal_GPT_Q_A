# -*- coding: utf-8 -*-
"""
Created on 15/02/2023
@author: gw.kayak
"""
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, BertModel, BertConfig, GPT2Config
from transformers import TextGenerationPipeline, AutoTokenizer, AutoModelWithLMHead
from transformers import AutoConfig, AutoModelWithLMHead

from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import torch.nn.functional as F
from loguru import logger




class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)




class Multimodal_GPT(nn.Module):

    def __init__(self,  prefix_len=10, clip_size=512,  constant_len=10,finetune_gpt2=False):
        super(Multimodal_GPT, self).__init__()
        self.finetune_gpt2 = finetune_gpt2
        self.gpt2=GPT2LMHeadModel.from_pretrained('H:/GPT_2_MODEL/gpt_chinese_small/pytorch_model.bin',config='H:/GPT_2_MODEL/gpt_chinese_small/config.json')
        self.prefix_size = 768
        self.prefix_len = 10
        self.clip_project = MLP((clip_size, (self.prefix_size * prefix_len)))

    def forward(self, clip_embeds, caption_ids, answer_ids,mask):
        """

        :param clip_embeds: 图像embedding, [bs, clip_size]
        :param caption_ids: caption的文本id, [bs, len]
        :param mask: 对于caption的文本id的attention mask, [bs, len]
        :return:
        """
        # caption_inputs_embeds:[bs, caption_len, prefix_size]
        caption_embeds = self.gpt2.transformer.wte(caption_ids)
        answer_embeds = self.gpt2.transformer.wte(answer_ids)
        # prefix_embeds:[bs, prefix_len, prefix_size]
        prefix_embeds = self.clip_project(clip_embeds).view(-1, self.prefix_len, self.prefix_size)
        # torch.Size([1, 10, 768]) torch.Size([1, 40, 768]) torch.Size([1, 50, 768])
        # embedding_cat:[bs, prefix_len+caption_len, prefix_size]
        embedding_cat = torch.cat((prefix_embeds, caption_embeds,answer_embeds), dim=1)
        #print(embedding_cat.shape)
        out = self.gpt2(inputs_embeds=embedding_cat,attention_mask=mask)
        # logits:[bs, prefix_len+caption_len, prefix_size]
        logits = out.logits
        #answer_logits = logits[:, -answer_ids.size(1):, :]
        
        return logits

    def parameters(self, recurse: bool = True):
        if self.finetune_gpt2:
            return super(ClipCaptionModel, self).parameters()
        else:
            return self.clip_project.parameters()

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.finetune_gpt2:
            self.gpt2.eval()
        return self
