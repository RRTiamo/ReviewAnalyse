import time

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_daataloader
from tokenizer import JiebaTokenizer
from model import ReviewAnalyzeModel
import config


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for review, label in tqdm(dataloader, desc='训练'):
        review = review.to(device)
        label = label.to(device)
        # 前向传播
        output = model(review)
        # 计算损失
        loss = loss_fn(output, label)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train():
    print("训练开始")
    # 获取设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 获取数据
    dataloader = get_daataloader()
    # 创建词表对象
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESS_DIR / 'vocab.txt')
    # 准备模型
    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size, padding_index=tokenizer.pad_token_id).to(device)
    # 准备优化器
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # 准备损失函数
    loss_fn = nn.BCEWithLogitsLoss()
    # 可视化训练
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d-%H-%M-%S"))
    # 保存模型参数
    best_loss = float('inf')
    # 开始训练
    for epoch in range(1, config.EPOCHS + 1):
        print(f"========== EPOCH{epoch} ==========")
        # 开启一轮训练
        avg_loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(avg_loss)
        writer.add_scalar("Loss", avg_loss, epoch)
        # 保存模型参数
        if best_loss > avg_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print("模型保存成功")
        else:
            print("模型未保存")
    print("训练结束")


if __name__ == '__main__':
    train()
