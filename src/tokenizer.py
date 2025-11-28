import jieba
from tqdm import tqdm

import config


class JiebaTokenizer:
    """
    自定义分词器
    """
    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(self, vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)

        self.word2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2word = {index: word for index, word in enumerate(vocab_list)}

        self.unk_token_id = self.word2index[self.unk_token]
        self.pad_token_id = self.word2index[self.pad_token]

    @staticmethod
    def _tokenizer(sentence):  # 分词 变为私有方法
        return jieba.lcut(sentence)

    # 大于seq_len截断，小于seq_len填充
    def encode(self, sentence, seq_len):  # 转换为index_list
        word_list = self._tokenizer(sentence)
        if len(word_list) > seq_len:
            # 截断
            word_list = word_list[0:seq_len]
        elif len(word_list) < seq_len:
            # 在后面填充
            word_list = word_list + [self.pad_token] * (seq_len - len(word_list))
        return [self.word2index.get(word, self.unk_token_id) for word in word_list]

    @classmethod
    def from_vocab(cls, vocab_file_path):  # 创建对象方法
        # 加载词表
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]

        return cls(vocab_list)

    # 构建词表
    @classmethod
    def build_vocab(cls, sentence, vocab_file_path):
        # 基于训练集构建词表映射，id到词
        vocab = set()
        for sentence in tqdm(sentence, desc='构建词表'):
            for word in jieba.lcut(sentence):
                if word.strip() != '':  # 去除不可见字符（空格和\t）
                    vocab.add(word)

        # 处理未登录词unk和填充词pad
        vocab_list = [cls.pad_token, cls.unk_token] + list(vocab)
        # 保存词表
        with open(vocab_file_path, 'w', encoding='utf8') as f:
            for word in vocab_list:
                f.write(word + '\n')
        print("词表保存完成")


if __name__ == '__main__':
    vocab = JiebaTokenizer.from_vocab(config.PROCESS_DIR / 'vocab.txt')
    index_list = vocab.encode('我喜欢坐地铁')
    print(index_list)
