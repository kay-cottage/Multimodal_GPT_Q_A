# -*- coding: utf-8 -*-
"""
Created on 15/02/2023
@author: gw.kayak
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertTokenizer
from tqdm import tqdm
import os
import argparse
from my_dataset import MyDataset
from model.model import Multimodal_GPT
import time
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from os.path import join
import transformers
import torch.nn.functional as F


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/clip_caption.pkl')
    parser.add_argument('--gpt2_path', default='pretrain_models/gpt2')
    parser.add_argument('--bert_path', default='pretrain_models/bert')
    parser.add_argument('--output_path', default='output')
    parser.add_argument("--lr", type=float, default=5e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--prefix_len', type=int, default=10)
    parser.add_argument('--constant_len', type=int, default=10)
    parser.add_argument('--clip_size', type=int, default=512)
    parser.add_argument('--bs_train', type=int, default=2)
    parser.add_argument('--dev_size', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--finetune_gpt2', help='finetune gpt2', action='store_true', default=False)
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument("--do_train", action='store_true', default=True)
    # parser.add_argument("--do_test", action='store_true', default=True)
    args = parser.parse_args()
    return args

# train demo
def train(model, train_loader,  optimizer,  args):
    model.train()
    logger.info("start training")
    device = args.device
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        print('epoch{}--------loss{}'.format(epoch+1,loss))
        for batch_idx, data in enumerate(tqdm(train_loader)):
            step = epoch * len(train_loader) + batch_idx + 1
            clip_embeds, caption_ids, answer_ids,mask = data
            clip_embeds = clip_embeds.to(device).float()
            caption_ids = caption_ids.to(device)
            answer_ids = answer_ids.to(device)
            mask = mask.to(device)
            logits = model(clip_embeds, caption_ids, answer_ids,mask)

            # 计算loss
            shift_logits = logits[:, -answer_ids.size(1):, :].contiguous().view(-1, logits.size(-1))
            #shift_logits = logits.contiguous().view(-1, logits.size(-1))
            shift_labels = answer_ids.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    torch.save(model.state_dict(), 'model/1.pt')


def evaluate(args, model, dataloader):
    """
    计算数据集上的指标
    :return:
    """
    model.eval()
    device = args.device
    logger.info("Running evaluation")
    eval_loss = 0.0  #
    with torch.no_grad():
        for data in tqdm(dataloader):
            clip_embeds, caption_ids, mask = data
            clip_embeds = clip_embeds.to(device).float()
            caption_ids = caption_ids.to(device)
            mask = mask.to(device)
            logits = model(clip_embeds, caption_ids, mask)

            # 计算loss
            shift_logits = logits[..., args.prefix_len+len(caption_ids) - 1:-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = caption_ids.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)

            loss = loss.mean()  # 对多卡的loss取平均
            eval_loss += loss
    return eval_loss


def main(args):

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载模型
    model = Multimodal_GPT().to(args.device)
    model.load_state_dict(torch.load('model/1.pt'))

    # 导入数据集
    dataset =
    [('id/1.png','caption/question 1','answer 1'),
     ('id/2.png','caption/question 2','answer 2'),
     ('id/3.png','caption/question 3','answer 3'),
     ('id/N.png','caption/question N','answer N')]

    dataset = MyDataset(dataset)
    train_dataloader = DataLoader(dataset, batch_size=args.bs_train, shuffle=True, num_workers=args.num_workers)
    optimizer = transformers.AdamW(model.parameters(), lr=5e-5)
    train(model, train_dataloader,  optimizer,  args)


if __name__ == '__main__':
    args = set_args()
    main(args)
