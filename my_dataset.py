
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
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from os.path import join
from loguru import logger
import glob
import skimage.io as io
from PIL import Image


class MyDataset(Dataset):
    """
                    data_list demo

    [('id/1.png','caption/question 1','answer 1'),
    ('id/2.png','caption/question 2','answer 2'),
    ('id/3.png','caption/question 3','answer 3')
                        ...
    ('id/N.png','caption/question N','answer N')]

    param:  prefix_len 默认10
    param:  max_len    默认50
    param:  dataset/img_data.pkl  放入经过clip(VIT-B16)处理的图片张量列表 [b_s,512]
    param:  pretrain_models/gpt2  tokenizer   
    """
    def __init__(self, data_list,prefix_len=10,  max_len=100,  normalize_prefix=False):
        self.normalize_prefix = normalize_prefix
        self.clip_embeds=torch.load('dataset/img_data.pkl')['img']
        tokenizer = BertTokenizer.from_pretrained('pretrain_models/gpt2')
        pad_id = tokenizer.pad_token_id

        clip_embeds = []
        caption_ids_list = []
        mask_list = []
        for img_id,caption, answer in data_list:
            clip_embed =self.clip_embeds[img_id].squeeze(0).float()
            caption_ids = tokenizer.encode(caption, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer, add_special_tokens=False)

            caption_ids.append(tokenizer.sep_token_id)
            caption_ids+=answer_ids
            caption_ids.append(tokenizer.sep_token_id)
            

            caption_ids = caption_ids[:max_len-prefix_len]
            mask = [1] * (prefix_len + len(caption_ids))

                # padding
            padding_len = max_len - prefix_len - len(caption_ids)
            caption_ids += [pad_id]*padding_len
            mask += [0]*padding_len


            caption_ids = torch.tensor(caption_ids).long()
            mask = torch.tensor(mask).long()



            clip_embeds.append(clip_embed)
            caption_ids_list.append(caption_ids)
            mask_list.append(mask)

        self.clip_embeds = clip_embeds
        self.caption_ids_list = caption_ids_list
        self.mask_list = mask_list
        
    def __len__(self) -> int:
        return len(self.caption_ids_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        clip_embed = self.clip_embeds[index]
        caption_ids = self.caption_ids_list[index]
        mask = self.mask_list[index]
        if self.normalize_prefix:
            clip_embed =clip_embed / clip_embed.norm(2, -1)
        return clip_embed, caption_ids, mask




