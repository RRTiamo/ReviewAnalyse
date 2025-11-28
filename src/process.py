import pandas as pd
from sklearn.model_selection import train_test_split

from tokenizer import JiebaTokenizer

import config


def process():
    print("数据预处理开始")
    # 读取数据
    df = pd.read_csv(config.RAW_DIR / 'online_shopping_10_cats.csv', usecols=['label', 'review'], encoding='utf-8')
    # 预处理
    df = df.dropna()
    # 划分训练集和测试集 stratify 防止取到的测试机都是1
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    # 构建词表
    JiebaTokenizer.build_vocab(train_df['review'].tolist(), config.PROCESS_DIR / 'vocab.txt')
    # 构建分词器对象
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESS_DIR / 'vocab.txt')

    # 构建训练集
    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))
    # 截断和填充的长度为128，95%的分位数
    # print(train_df['review'].apply(lambda x: len(x)).quantile(0.95))
    # 保存训练集
    train_df.to_json(config.PROCESS_DIR / 'index_train.jsonl', orient='records', lines=True)

    # 构建测试集
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x, config.SEQ_LEN))
    # 保存测试集
    test_df.to_json(config.PROCESS_DIR / 'index_test.jsonl', orient='records', lines=True)
    print("数据预处理结束")


if __name__ == '__main__':
    process()
