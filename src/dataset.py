import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class ReviewAnalyzeDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, lines=True, orient='records').to_dict(
            orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['review'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['label'], dtype=torch.float32)
        return input_tensor, target_tensor


def get_daataloader(train=True):
    path = 'index_train.jsonl' if train else 'index_test.jsonl'
    dataset = ReviewAnalyzeDataset(config.PROCESS_DIR / path)  # 数据对象
    # DataLoader 的第一个参数必须是Dataset数据类型的数据
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    dataloader = get_daataloader()
    for input_tensor, target_tensor in dataloader:
        print(input_tensor.shape)  # torch.Size([128, 128])
        print(target_tensor.shape)  # torch.Size([128])
        print(len(dataloader))
        break
