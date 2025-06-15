import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
# from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn
import datetime
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split

class CNN(nn.Module):
    def __init__(
        self,
        in_channels = 3,
        hidden_channels = (32, 64, 128, 256),
        num_classes = 5,
        kernel_size = 3,
        dropout = 0.1,
        adaptiveavg = 4,
        input_dim = 4 * 4,
        hidden_dim = 512
        ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size, padding=1),#彩色图像输入3通道
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size, padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size, padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            
            nn.Conv2d(hidden_channels[2], hidden_channels[3], kernel_size, padding=1),
            nn.BatchNorm2d(hidden_channels[3]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        
            nn.AdaptiveAvgPool2d((adaptiveavg, adaptiveavg)) 
        )
        
        self.classifier = nn.Sequential( 
            nn.Linear(hidden_channels[3] * input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))
    
    def set_device(self, device: torch.device):
        self.to(device)
        self.device = device
        return self

class MLP(nn.Module):
    def __init__(
        self,
        input_dim = 3 * 224 * 224,
        hidden_dim = (512, 128, 32, 5),
        output_dim = 5,
        dropout = (0.5, 0.3, 0.2)
        ):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(dropout[0]),
            
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Dropout(dropout[1]),
            
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Dropout(dropout[2]),
            nn.Linear(hidden_dim[2], hidden_dim[3])
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
    def set_device(self, device: torch.device):
        self.to(device)
        self.device = device
        return self