import torch
from torch import nn
from torchinfo import summary

import config


class ReviewAnalyzeModel(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        # # 多层双向
        self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True,
                            bidirectional=True, num_layers=2)
        # # 多层单向
        # self.lstm = nn.LSTM(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_SIZE, batch_first=True,
        #                     num_layers=2)

        # 解决维度不一致: (128*512) * (512,1) = 128*1
        self.linear = nn.Linear(in_features=2 * config.HIDDEN_SIZE, out_features=1)  # 多层多项
        # self.linear = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=1) # 多层单向

    def forward(self, X):
        embed = self.embedding(X)  # (batch_size,seq_len,embedding_dim)
        # 避免验证时输入batch_size不匹配
        batch_size = X.shape[0]
        ## 多层单向
        # h0 = torch.randn((2, batch_size, config.HIDDEN_SIZE)).to('cuda')
        ## 多层双向
        h0 = torch.randn((2 * 2, batch_size, config.HIDDEN_SIZE)).to('cuda')
        c0 = torch.randn_like(h0).to('cuda')
        output, (hn, _) = self.lstm(embed, (h0, c0))  # (batch_size,seq_len,hidden_size)
        #################################### 多层双向 ##############################################
        # 正向最后一层：hn[-2, :, :]；反向最后一层：hn[-1, :, :]
        # hn形状：(num_layers * num_directions, batch_size, hidden_size)
        h1 = hn[-2, :, :]
        hn = hn[-1, :, :]
        # 按列拼接[h1,hn]
        result_stack = torch.cat([h1, hn], dim=1)
        # result_stack.shape  # torch.Size([128, 512])
        # hn (1,batch_size,seq_len)
        output = self.linear(result_stack)
        #########################################################################################
        # last_hidden = output[:, -1, :]  # (128*256)
        # output = self.linear(last_hidden)
        output.squeeze_(1)  # 这里保证维度一致
        return output


if __name__ == '__main__':
    # 打印模型摘要

    model = ReviewAnalyzeModel(vocab_size=20000, padding_index=0).to('cuda')
    dummy_input = torch.randint(
        low=0,
        high=20000,
        size=(config.BATCH_SIZE, config.SEQ_LEN),
        dtype=torch.long,
        device='cuda'
    )
    # summary(model, input_data=dummy_input)
    print(model(dummy_input).shape)
# ==========================================================================================
# ReviewAnalyzeModel                       [128, 1]                  --
# ├─Embedding: 1-1                         [128, 128, 128]           2,560,000
# ├─LSTM: 1-2                              [128, 128, 512]           2,367,488
# ├─Linear: 1-3                            [128, 1]                  513
# ==========================================================================================
