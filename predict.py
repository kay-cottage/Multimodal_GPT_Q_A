
# -*- coding: utf-8 -*-
"""
Created on 15/02/2023
@author: gw.kayak
"""
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer
from tqdm import tqdm
import os
import argparse
from model.model import Multimodal_GPT
from loguru import logger
from os.path import join
import torch.nn.functional as F
from PIL import Image
import clip



def topk_filtering(logits, topk=3, topp=0, filter_value=-float('Inf')):
    # todo topp
    """
    将topk以外的token的生成概率置为-inf
    :param logits: [b_size, dim]
    :param topk:
    :param filter_value:
    :return:
    """
    assert logits.dim() == 2  # batch size 1 for now - could be updated for more but the code would be less clear
    topk = min(topk, logits.size(-1))  # Safety check
    if topk > 0:
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        indices_to_remove = logits < torch.topk(logits, topk, dim=-1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if topp > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > topp
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # todo check
        for i in range(sorted_indices_to_remove.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def generate(model, clip_embeds, caption,tokenizer):
    """

    :param model:
    :param clip_embeds: [b_size x clip_size]
    :param temperature 
    :param topk
    :param topp
    :param max_len   
    :param device
    :return:
    """
    b_size = clip_embeds.size(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pad_id = tokenizer.pad_token_id
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id
    max_len = 50
    temperature = 1
    topk = 1
    topp = 0.8
    cur_len = 0
    caption_ids = []    # 存储生成的caption



    caption = tokenizer.encode(caption, add_special_tokens=False)
    caption.append(tokenizer.sep_token_id)
    caption=torch.tensor([caption]).long()
        
    # gpt2模型的输入: inputs_embeds:[bs, prefix_len, prefix_size]
    # inputs_embeds = model.clip_project(clip_embeds).view(-1, model.prefix_len, model.prefix_size)
    prefix_embeds = model.clip_project(clip_embeds).view(-1, 10, 768)
    caption_embeds = model.gpt2.transformer.wte(caption)
    inputs_embeds = torch.cat((prefix_embeds, caption_embeds), dim=1)
    finish_flag = [False] * b_size  # 第i个输入是否完成生成的标志

    while True:
        out = model.gpt2(inputs_embeds=inputs_embeds)
        logits = out.logits  # [b_size, len, vocab_size]
        next_token_logits = logits[:, -1, :]    # 取最后一个单词的预测分布
        next_token_logits = next_token_logits / temperature
        next_token_logits[:, unk_id] = -float('Inf')   # 将unk设为无穷小

        # topk filter
        filtered_logits = topk_filtering(next_token_logits, topk, topp)
        next_token_ids = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).squeeze(1).tolist()

        # 分别判断生成图片是否已生成完毕
        for index in range(len(next_token_ids)):
            token_id = next_token_ids[index]
            # 如果第i个句子已经生成结束
            if finish_flag[index]:
                next_token_ids[index] = pad_id
            # 如果第i个句子生成结束
            elif token_id == sep_id:
                finish_flag[index] = True
            # 未结束生成
            elif cur_len == 0:
                caption_ids.append([token_id])
            else:
                caption_ids[index].append(token_id)
        next_token_ids = torch.tensor(next_token_ids).to(device)
        next_token_embeds = model.gpt2.transformer.wte(next_token_ids).to(device).unsqueeze(1)
        inputs_embeds = torch.cat((inputs_embeds, next_token_embeds), dim=1)

        cur_len += 1
        if cur_len > max_len or False not in finish_flag:
            break

    # 对token_id进行解码
    captions = []
    for caption_id in caption_ids:
        caption = tokenizer.convert_ids_to_tokens(caption_id)
        caption = ''.join(caption)
        captions.append(caption)

    return captions

# load clip model to image vector
def get_image_tensor(clip_embeds_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(clip_embeds_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features
    



def main(use_clip=False):
    
    # 分词器
    model = Multimodal_GPT().to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # 加载预训练权重
    model.load_state_dict(torch.load('pretrain_models/1.pt'))
    model.eval()
    # 分词器
    tokenizer = BertTokenizer.from_pretrained('pretrain_models/gpt2')

    # 可以用clip把需要用到的图片提前编码存入字典中以减少运行所需的内存
    # clip_embeds  {'path_1':tensor_1,'path_2':tensor_2,...}
    clip_embeds_dict=torch.load('dataset/img.pkl')

    while 1:
        clip_embeds_path=input('Image Path:')
        caption=input('Say something Or Ask something:')
        if not use_clip:
            # 可以用clip把需要用到的图片提前编码存入字典中以减少运行所需的内存
            # clip_embeds  {'path_1':tensor_1,'path_2':tensor_2,...}
            clip_embeds_dict=torch.load('dataset/img.pkl')
            
            image=Image.open(clip_embeds_path)
            image.show()
            clip_embeds=clip_embeds_dict[clip_embeds_path].squeeze(0).float()
            answer=generate(model, clip_embeds, caption,tokenizer)
            print(answer)
        else:
            image=Image.open(clip_embeds_path)
            image.show()
            clip_embeds=get_image_tensor(clip_embeds_path).squeeze(0).float()
            answer=generate(model, clip_embeds, caption,tokenizer)
            print(answer)
            

if __name__ == '__main__':
    main()
