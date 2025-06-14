import torch
from torch import optim
from torch.nn.functional import softmax
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import logging
from time import time
import os

from model import CNN
from model import MLP
from dataset import TCMDataset
from logging import FileHandler
import csv

import matplotlib
matplotlib.use('Agg')  # 无GUI环境必备
import matplotlib.pyplot as plt

import datetime
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser()

# model hyperparameters
parser.add_argument('--input_dim', type=int, default=28*28, help='input dimension')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
parser.add_argument('--out_dim', type=int, default=10, help='output dimension')
parser.add_argument('--num_layer', type=int, default=2, help='number of hidden layers')
parser.add_argument('--activation', type=str, default='sigmoid', help='activation function')

# dataset parameters
parser.add_argument('--data_path', type=str, default='./data/mnist/', help='path to dataset')

# train parameters
parser.add_argument('--batch_size', type=int, default=64, help='batch size, default: 64')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate, default: 0.1')
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu', help='device')
parser.add_argument('--optimizer', type=str, default='SGD',\
                    help='set optimizer type in [SGD/Adam/AdamW...], default: SGD')
parser.add_argument('--lr', type=float, default=1e-3,\
                    help='set learning rate, default: 1e-3')
parser.add_argument('--epoch', type=int, default=2, help='epoch to train, default: 10')
parser.add_argument('--patience', type=int, default=20, help='epoch to stop training, default: 20')
parser.add_argument('--log_file', type=str, default='log/log.txt',\
                    help='log file name, default: log/log.txt')
parser.add_argument('--save_path', type=str, default='model/best.pt',\
                    help='path to save model, default: model/best.pt')
parser.add_argument('--loss_img', type=str, default='log/loss.png', help='path to save loss image, default: log/loss.png')

args = parser.parse_args()
def train(model, type):
    
    if not os.path.exists('/'.join(args.log_file.split('/')[:-1])):
        os.makedirs('/'.join(args.log_file.split('/')[:-1]))
    if not os.path.exists('/'.join(args.save_path.split('/')[:-1])):
        os.makedirs('/'.join(args.save_path.split('/')[:-1]))

    # 创建 FileHandler 并指定编码
    file_handler = FileHandler(
        filename=args.log_file,  # 使用传入的日志文件路径
        mode='w',                # 覆盖模式（原 filemode='w'）
        encoding='utf-8'         # 指定 UTF-8 编码
    )

    # 正确配置 logging
    logging.basicConfig(
        format="[%(asctime)s]:%(levelname)s: %(message)s",
        datefmt="%d-%M-%Y %H:%M:%S",
        level=logging.DEBUG,
        handlers=[file_handler]  # 只通过 handlers 传递
    )
    
    torch.manual_seed(time())

    train_set = TCMDataset(root_dir='data/tcm/train')
    test_set = TCMDataset(root_dir='data/tcm/test')
    
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True
    )
        
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
        
    device = torch.device(args.device)
    
    model = model.set_device(device)
    
    # claim the loss function
    criterion = torch.nn.CrossEntropyLoss()     # 交叉熵损失衡量
    
    # claim the optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    start = 1

    train_acc = []
    train_f1 = []
    train_loss = []
    
    model.train() # 设置为训练模式，使得dropout生效
    for epoch in tqdm(range(start, args.epoch + 1), desc="[Total Training]", position=0):
        logging.info('[epoch: {:d}] '.format(epoch))
        
        truth = []
        predict = []
        total_loss = 0.
        
        for pack_images, y, _ in train_loader:
            pack_images = pack_images.to(device)
            y = y.to(device)
            
            truth = truth + y.tolist()
            
            # forward and backward
            optimizer.zero_grad()          # 清零梯度
            output = model(pack_images)     # 前向计算 [batch_size, out_dim]
            loss = criterion(output, y)    # 计算损失
            loss.backward()                # 反向传播
            optimizer.step()               # 更新参数
            
            # 记录预测结果和损失
            predict += output.argmax(dim=1).tolist()  # 取概率最大的类别作为预测结果
            total_loss += loss.item()      # 累加损失值
            
        logging.info('average loss: {:.4f}'.format(total_loss * args.batch_size / len(train_loader)))
        train_loss.append(total_loss / len(train_loader))
        
        acc = accuracy_score(truth, predict)
        marco_f1 = f1_score(truth, predict, average='macro')
        precision = precision_score(truth, predict, average='macro')
        recall = recall_score(truth, predict, average='macro')
        
        logging.info('train acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
            acc,
            marco_f1, 
            precision, 
            recall)
        )
        train_acc.append(acc)
        train_f1.append(marco_f1)
        
    truth = []
    predict = []
    filenames_list = []  
    logging.info('test model on test set')

    model.eval()
    for pack_images, y, filenames in tqdm(test_loader, desc="[test on test set] ", leave=False):  # NEW: 解包文件名
        with torch.no_grad():
            pack_images = pack_images.to(device)
            y = y.to(device)
            
            truth = truth + y.tolist()
            filenames_list += filenames  

            output = model(pack_images)
            loss = criterion(output, y)
            
            predict += output.argmax(dim=1).tolist()

    results = []
    for filename, true_label, pred_label in zip(filenames_list, truth, predict):
        results.append({
            "filename": filename,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": (true_label == pred_label)
        })
        print(f"文件: {filename} | 真实标签: {true_label} | 预测标签: {pred_label}")
        

    save_dir = "TCM_training_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{type}_test_results.csv")
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_label", "pred_label", "correct"])
        writer.writeheader()
        writer.writerows(results)
    
    return train_loss, train_acc, train_f1

if __name__ == "__main__":
    cnn = CNN()
    mlp_model = MLP()
    cnn_loss, cnn_acc, cnn_f1 = train(cnn, 'cnn')
    mlp_loss, mlp_acc, mlp_f1 = train(mlp_model, 'mlp')
    # -----------------------------可视化及保存------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.title("TCM training Loss")

    # sigma = 1.2
    # smooth_cnn_loss = gaussian_filter1d(cnn_loss, sigma=sigma)
    # smooth_mlp_loss = gaussian_filter1d(mlp_loss, sigma=sigma)
    # smooth_cnn_acc = gaussian_filter1d(cnn_acc, sigma=sigma)
    # smooth_mlp_acc = gaussian_filter1d(mlp_acc, sigma=sigma)

    plt.plot(cnn_loss, 'b-', label='CNN Loss', alpha=0.8)
    plt.plot(mlp_loss, 'r--', label='MLP Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tcm_comparison.png"
    save_dir = "TCM_training_results"
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录
    save_path = os.path.join(save_dir, filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {os.path.abspath(save_path)}")