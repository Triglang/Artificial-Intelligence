from torch.utils.data import Dataset
import torch
import os
from matplotlib.image import imread
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm

class Mnist1DDataset(Dataset):
    def __init__(self, data_path:str, set_type = 'train'):
        super().__init__()
        self.images = []
        self.labels = []
        self.set_type = set_type
        self.max_lenth = 0

        for file in tqdm(os.listdir(data_path), desc=f"Loading {set_type} Data"):
            img_path = os.path.join(data_path, file)
            file_name = file.split('.')[0]
            st = file_name.split('_')[0]
            if st == 'training':
                st = 'train'
            if st != self.set_type:
                continue

            self.labels.append(int(file_name.split('_')[2]))
            img = imread(img_path)
            img = img / 255.0 # 转换成0-1的灰度图
            self.max_lenth = max(self.max_lenth, np.array(list(img.shape)).prod())
            self.images.append(img)
        self.category = sorted(list(set(self.labels)))

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def collate_fn(batch):
        '''
        将取出的batch，图片转换到一维，再组织成tensor
        Input:
            - batch: [(img, label), (img, label), ...]
            Output:
                imgs: tensor[batch_size, max_lenth]
                labels: tensor[batch_size]
        '''
        images = []
        labels = []
        for item in batch:
            img = torch.tensor(item[0], dtype=torch.float)
            img = img.view(-1) # 将图片拉成一维
            images.append(img)
            labels.append(item[1])
        
        # 使用pad_sequence处理变长序列（假设图像长度不同，可适应）
        padded_images = pad_sequence(images, batch_first=True) # batch_first=True表示batch_size为第一维度
        labels = torch.tensor(labels, dtype=torch.long)
        return padded_images, labels
    
class housePriceDataset(Dataset):
    def __init__(self, data_path:str):
        super().__init__()
        self.xs = []
        self.ys = []
        self.min_price = None
        self.max_price = None

        df = pd.read_csv(data_path)
        for index, row in df.iterrows():
            self.xs.append(row[:-1])
            self.ys.append(row[-1])
            if self.min_price is None or row[-1] < self.min_price:
                self.min_price = row[-1]
            if self.max_price is None or row[-1] > self.max_price:
                self.max_price = row[-1]

        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)

        # 将特征和标签归一化
        x_min = self.xs.min(axis = 0)
        x_max = self.xs.max(axis = 0)
        self.xs = (self.xs - x_min) / (x_max - x_min)
        self.ys = (self.ys - self.min_price) / (self.max_price - self.min_price)

        # 转换回list
        self.xs = self.xs.tolist()
        self.ys = self.ys.tolist()

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]
    
    def __len__(self):
        return len(self.xs)
    
    @staticmethod
    def collate_fn(batch):
        '''
        将取出的batch组织成floatTensor
        Input:
            - batch: [(img, label), (img, label), ...]
            Output:
                imgs: tensor[batch_size, max_lenth]
                labels: tensor[batch_size]
        '''
        xs = []
        ys = []
        for item in batch:
            xs.append(item[0])
            ys.append(item[1])
        
        xs = torch.tensor(xs, dtype=torch.float)
        ys = torch.tensor(ys, dtype=torch.float)
        return xs, ys