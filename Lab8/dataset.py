from torch.utils.data import Dataset
import torch
import os
from matplotlib.image import imread
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from tqdm import tqdm
import cv2

def tcm_preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std
    
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).float()

classes_name = ['baihe','dangshen','gouqi','huaihua','jinyinhua']
class TCMDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.filename = []
        
        for label, class_name in enumerate(classes_name):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(tcm_preprocess_image(img_path))
                self.labels.append(label)
                self.filename.append(img_name)
                
        self.category = sorted(list(set(self.labels)))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.filename[idx]

def mnist_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.1307) / 0.3081  # MNIST标准归一化
    img = np.expand_dims(img, axis=0)  # 添加通道维度
    return torch.from_numpy(img).float()

class MNISTDataset(Dataset):
    def __init__(self, root_dir, mode='training'):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.filename = []
        
        for img_name in os.listdir(root_dir):
            if not img_name.endswith('.jpg'):
                continue
            parts = img_name.split('_')
            if len(parts) < 3 or parts[0] != mode:
                continue
            label = int(parts[2].split('.')[0])
            img_path = os.path.join(root_dir, img_name)
            self.images.append(mnist_preprocess_image(img_path))
            self.labels.append(label)
            self.filename.append(img_name)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.filename[idx]