import torch
from model import ReviewAnalyzeModel
from src.dataset import get_daataloader
from tokenizer import JiebaTokenizer
import config


# 批量预测
def batch_predict(input_list, model):
    with torch.no_grad():
        output = model(input_list)  # (seq_len)
        return torch.sigmoid(output).tolist()


def predict(model, user_input, tokenizer, device):
    model.eval()
    input_list = tokenizer.encode(user_input, config.SEQ_LEN)  # (batch_size,seq_len)
    # 转为张量进行计算
    input_list = torch.tensor([input_list]).to(device)
    batch_predict(input_list, model)
    return batch_predict(input_list, model)[0]


def run_predict():
    # print("开始测试")
    # 准备资源
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 准备词表
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESS_DIR / 'vocab.txt')
    # 准备模型
    model = ReviewAnalyzeModel(tokenizer.vocab_size, tokenizer.pad_token_id).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    print("请输入评论：按「q，quit」退出")
    while True:
        user_input = input(">")
        if user_input in ['q', 'quit']:
            break
        if user_input.strip() == '':
            continue
        res = predict(model, user_input, tokenizer, device)
        if res > 0.5:
            print(f'正面评价：置信度{res:.4f}')
        else:
            print(f'反面评价：置信度{1 - res:.4f}')


if __name__ == '__main__':
    run_predict()
