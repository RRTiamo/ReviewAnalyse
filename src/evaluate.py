import torch

from dataset import get_daataloader
from tokenizer import JiebaTokenizer
import config
from model import ReviewAnalyzeModel
from predict import batch_predict


def run_evaluate(dataloader, model, device):
    model.eval()
    total_count = 0
    total_acc = 0
    for review, label in dataloader:
        review = review.to(device)
        label = label.tolist()
        output_list = batch_predict(review, model)
        # 计算准确率
        for out, target in zip(output_list, label):
            # 概率值转换为0，1
            out = 1 if out > 0.5 else 0
            if out == target:
                total_count += 1
                total_acc += 1
            else:
                total_count += 1
    return total_acc / total_count


def evaluate():
    # print("开始测试")
    # 准备资源
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 准备词表
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESS_DIR / 'vocab.txt')
    # 准备模型
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_id).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    # 准备数据
    dataloader = get_daataloader(False)
    acc = run_evaluate(dataloader, model, device)
    print("========== 评估结果 ==========")
    print(f"acc{acc:.4f}")
    print("=============================")


if __name__ == '__main__':
    evaluate()
