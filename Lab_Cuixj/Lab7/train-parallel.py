import torch
from torch import optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, random_split
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import logging
from time import time
import os

from mlp import MLP
from dataset import Mnist1DDataset

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
parser.add_argument('--epoch', type=int, default=400, help='epoch to train, default: 400')
parser.add_argument('--patience', type=int, default=20, help='epoch to stop training, default: 20')
parser.add_argument('--log_file', type=str, default='log/log.txt',\
                    help='log file name, default: log/log.txt')
parser.add_argument('--save_path', type=str, default='model/best.pt',\
                    help='path to save model, default: model/best.pt')

parser.add_argument('--num_workers', type=int, default=1, help='数据加载进程数')
parser.add_argument('--cpu_threads', type=int, default=1, help='CPU并行计算线程数')
parser.add_argument('--pin_memory', type=int, choices=[0,1], default=1, help='是否启用内存锁页加速 (0/1)')

DEBUG = 0

def main():       
    args = parser.parse_args()
    if DEBUG:
        print(args)
    
    if not os.path.exists('/'.join(args.log_file.split('/')[:-1])):
        os.makedirs('/'.join(args.log_file.split('/')[:-1]))
    if not os.path.exists('/'.join(args.save_path.split('/')[:-1])):
        os.makedirs('/'.join(args.save_path.split('/')[:-1]))
    logging.basicConfig(
        filename=args.log_file, 
        filemode="w", 
        format="[%(asctime)s]:%(levelname)s: %(message)s", 
        datefmt="%d-%M-%Y %H:%M:%S", 
        level=logging.DEBUG
    )
    
    PARALLEL_CONFIG = {
        # 数据加载
        'DARA_LOADER': args.num_workers,       # 训练集数据加载进程数
        'PIN_MEMORY': False,       # 是否启用内存锁页加速
        
        # CPU计算
        'CPU_THREADS': args.cpu_threads,         # CPU并行计算线程数
        'OMP_NUM_THREADS': args.cpu_threads,     # OpenMP线程数
        'MKL_NUM_THREADS': args.cpu_threads,     # MKL数学库线程数
    }
    
    # 配置CPU并行参数
    torch.set_num_threads(PARALLEL_CONFIG['CPU_THREADS'])
    os.environ['OMP_NUM_THREADS'] = str(PARALLEL_CONFIG['OMP_NUM_THREADS'])
    os.environ['MKL_NUM_THREADS'] = str(PARALLEL_CONFIG['MKL_NUM_THREADS'])
    
    # Windows系统特殊处理
    if os.name == 'nt':  
        torch.multiprocessing.set_start_method('spawn', force=True)

    # 获取数据集
    torch.manual_seed(time())
    if 'mnist' in args.data_path:
        data_set = Mnist1DDataset(args.data_path, set_type='train')
        test_set = Mnist1DDataset(args.data_path, set_type='test')

        train_size = data_set.__len__() // 5 * 4
        val_size = data_set.__len__() - train_size
        train_set, val_set = random_split(data_set, [train_size, val_size])
    
        train_loader = DataLoader(
            train_set, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=PARALLEL_CONFIG['DARA_LOADER'],
            pin_memory=PARALLEL_CONFIG['PIN_MEMORY'],
            collate_fn=Mnist1DDataset.collate_fn
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=PARALLEL_CONFIG['DARA_LOADER'],
            pin_memory=PARALLEL_CONFIG['PIN_MEMORY'],
            collate_fn=Mnist1DDataset.collate_fn
        )
        test_loader = DataLoader(
            test_set, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=PARALLEL_CONFIG['DARA_LOADER'],
            pin_memory=PARALLEL_CONFIG['PIN_MEMORY'],
            collate_fn=Mnist1DDataset.collate_fn
        )
    else:
        raise ValueError('Invalid dataset path')

    category = data_set.category
    
    device = torch.device(args.device)
    # 声明模型、损失函数、优化器
    ################################################
    # TODO：
    #       claim the model and set your model parameter here
    # hint:
    #       claim the model with variable name "model" or rewrite the code below
    model = MLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        device=device,
        # 如果 MLP 类支持以下参数则取消注释
        # num_layers=args.num_layer,
        # activation=args.activation,
        # dropout=args.dropout
    )
    model.set_device(device)
    ################################################ 

    ################################################
    # TODO：
    #       claim the loss function here
    criterion = torch.nn.CrossEntropyLoss()     # 交叉熵损失衡量
    ################################################ 

    ################################################
    # TODO：
    #       claim the optimizer here
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    ################################################ 
    
    start = 1
    early_stop_count = 0
    best_val_f1 = 0.

    train_acc = []
    train_f1 = []
    train_loss = []

    val_acc = []
    val_f1 = []
    val_loss = []


    # for epoch in range(start, args.epoch + 1):
    epochs_bar = tqdm(range(start, args.epoch + 1), desc="Total Training", position=0)
    for epoch in epochs_bar:
        logging.info('[epoch: {:d}] '.format(epoch))

        # 训练模型
        truth = []
        predict = []
        total_loss = 0.
        
        # 在训练集上训练模型
        model.train() # 设置为训练模式，使得dropout生效
        # for pack_images, y in tqdm(train_loader, desc="[epoch: {:d}, training] ".format(epoch), leave=False):
        for pack_images, y in train_loader:
            #############################
            # TODO：
            #       put the data above to device
            pack_images = pack_images.to(device)
            y = y.to(device)
            #############################
            truth = truth + y.tolist()

            #############################
            # TODO：
            #       your forward and backward code here
            # hint:
            #       remember to maintain List: truth and predict
            #       or remove the metric calculation below
            optimizer.zero_grad()          # 清零梯度
            output = model(pack_images)     # 前向计算 [batch_size, out_dim]
            loss = criterion(output, y)    # 计算损失
            loss.backward()                # 反向传播
            optimizer.step()               # 更新参数

            # 记录预测结果和损失
            predict += output.argmax(dim=1).tolist()  # 取概率最大的类别作为预测结果
            total_loss += loss.item()      # 累加损失值
            if DEBUG:
                print(predict)
                print(truth)
                print(total_loss)
            #############################

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

        # 更新外层进度条显示
        epochs_bar.set_postfix({
            "train_acc": f"{acc:.2%}",
            "train_f1": f"{marco_f1:.4f}",
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # 在验证集上测试并保存最好的模型
        logging.info('======')
        logging.info('test model on val set')
        truth = []
        predict = []
        total_loss = 0.

        model.eval() # 设置为测试模式，使得dropout不生效
        for pack_images, y in val_loader:
            with torch.no_grad():  # 禁用梯度计算
            # for pack_images, y in tqdm(val_loader, desc="[epoch: {:d}, test on val set] ".format(epoch), leave=False):
                #############################
                # TODO：
                #       put the data above to device
                pack_images = pack_images.to(device)
                y = y.to(device)
                #############################
                truth = truth + y.tolist()

                #############################
                # TODO：
                #       your forward and backward code here
                # hint:
                #       remember to maintain List: truth and predict
                #       or remove the metric calculation below
                # 前向传播和损失计算
                output = model(pack_images)      # 前向计算
                loss = criterion(output, y)     # 计算损失
                # 记录结果
                predict += output.argmax(dim=1).tolist()
                total_loss += loss.item()
                #############################
        acc = accuracy_score(truth, predict)
        marco_f1 = f1_score(truth, predict, average='macro')
        precision = precision_score(truth, predict, average='macro')
        recall = recall_score(truth, predict, average='macro')

        logging.info('val loss: {:.4f}'.format(total_loss * args.batch_size / len(val_loader)))
        logging.info('val acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
            acc,
            marco_f1, 
            precision, 
            recall)
        )
        logging.info('early stop counter: {:d}'.format(early_stop_count))
        logging.info('========================================')
        val_loss.append(total_loss / len(val_loader))
        val_acc.append(acc)
        val_f1.append(marco_f1)

        #==============================================
        # save the best model or ignore this code
        if marco_f1 > best_val_f1:
            best_val_f1 = marco_f1
            # 保存模型参数
            torch.save(model.state_dict(), args.save_path)
            logging.info(f"🌟 发现新最佳模型，验证F1分数: {best_val_f1:.4f}")
            early_stop_count = 0  # 重置早停计数器
        else:
            early_stop_count += 1
            logging.warning(f"早停计数器: {early_stop_count}/{args.patience}")

        # 早停判断
        if early_stop_count >= args.patience:
            logging.warning(f"🚨 早停触发！连续 {args.patience} 轮验证F1未提升")
            break  # 终止训练循环
        #==============================================

        # 早停：验证集上连续多次性能测试没有提升，则停止训练
        ###############################################
        # set your early stop condition here
        # or ignore this code
        ###############################################
    
    # end train for loop

    #  在测试集上测试
    truth = []
    predict = []
    total_loss = 0.
    logging.info('test model on test set')

    model.eval()
    for pack_images, y in tqdm(test_loader, desc="[test on test set] ", leave=False):
        with torch.no_grad():
            #############################
            # TODO：
            #       put the data above to device
            pack_images = pack_images.to(device)
            y = y.to(device)
            #############################
            truth = truth + y.tolist()

            #############################
            # TODO：
            #       your forward and backward code here
            # hint:
            #       remember to maintain List: truth and predict
            #       or remove the metric calculation below
            output = model(pack_images)         # 前向计算
            loss = criterion(output, y)        # 计算损失
            
            # 记录结果
            predict += output.argmax(dim=1).tolist()
            total_loss += loss.item()
            #############################

    acc = accuracy_score(truth, predict)
    marco_f1 = f1_score(truth, predict, average='macro')
    precision = precision_score(truth, predict, average='macro')
    recall = recall_score(truth, predict, average='macro')
    logging.info('test loss: {:.4f}'.format(total_loss * args.batch_size / len(test_loader)))
    logging.info('test acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
        acc,
        marco_f1, 
        precision, 
        recall)
    )
    logging.info('classification report:')
    logging.info('\n' + str(classification_report(
        truth, predict, labels = category, digits = 4
    )))

# end main()

if __name__ == '__main__':
    main()