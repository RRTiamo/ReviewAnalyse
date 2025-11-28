from pathlib import Path

# __file__为当前文件的目录
ROOT_DIR = Path(__file__).parent.parent

# 目录
PROCESS_DIR = ROOT_DIR / 'data' / 'processed'
RAW_DIR = ROOT_DIR / 'data' / 'raw'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'

SEQ_LEN = 128  # 序列长度
BATCH_SIZE = 128  # 批量大小
EMBEDDING_DIM = 128  # 嵌入层维度
HIDDEN_SIZE = 256  # 隐藏层维度
LEARNING_RATE = 1e-3  # 学习率大小
EPOCHS = 10  # 训练轮次

if __name__ == '__main__':
    print(RAW_DIR)
