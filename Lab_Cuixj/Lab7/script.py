# run_experiments.py
import subprocess
import sys
from pathlib import Path
import logging
import os

def run_command(params):
    """执行单个训练命令，返回（是否成功，美化命令字符串）"""
    # 生成实际执行的命令列表
    cmd = [
        "python", r".\train.py",
        "--input_dim", str(params["input_dim"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--out_dim", str(params["out_dim"]),
        "--num_layer", str(params["num_layer"]),
        "--activation", str(params["activation"]),
        "--data_path", str(params["data_path"]),
        "--batch_size", str(params["batch_size"]),
        "--dropout", str(params["dropout"]),
        "--device", str(params["device"]),
        "--optimizer", str(params["optimizer"]),
        "--lr", str(params["lr"]),
        "--epoch", str(params["epoch"]),
        "--patience", str(params["patience"]),
        "--log_file", str(params["log_file"]),
        "--save_path", str(params["save_path"]),
        "--loss_img", str(params["loss_img"])
    ]
    
    # 生成美化后的命令字符串（带换行）
    pretty_cmd = " \\\n".join([
        "python .\\train.py"] + 
        [f"    --{arg.replace('_', '-')} {val}" 
         for arg, val in zip(cmd[2::2], cmd[3::2])]
    )
    
    print("\n" + "="*50)
    print(f"开始执行命令:\n{pretty_cmd}")
    print("="*50)

    log_dir = os.path.dirname(params["log_file"])
    cmd_path = os.path.join(log_dir, "cmd.txt")
    
    # 创建目录（如果不存在）
    os.makedirs(log_dir, exist_ok=True)
    
    # 直接写入文件（无需复杂logging配置）
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(f"执行命令:\n{pretty_cmd}\n")
    
    try:
        subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT
        )
        return True, pretty_cmd
    except subprocess.CalledProcessError as e:
        return False, pretty_cmd
    
def run_hd_nl_ep():
    pass

if __name__ == "__main__":
    input_dim = 28 * 28
    hidden_dim = [32, 64, 128]
    out_dim = 10
    num_layer = [2, 3, 4]
    
    activation = ['relu', 'sigmoid', 'tanh', 'softmax', 'none']
    
    data_path = r"./data/mnist_jpg"
    
    batch_size = 512
    dropout = [0.05, 0.1]   # 未测试
    device = "cpu"
    optimizer = ["sgd", "adam", "adamw"]
    lr = [0.001, 0.01, 1]
    epoch = [300, 400, 500]
    patience = 20
    log_file = ""
    save_path = ""
    loss_img = ""
    
    experiments = []  
    
    # 遍历隐藏层维度
    # 遍历层数
    # 遍历 epoch
    # for hd in hidden_dim:
    #     for nl in num_layer:
    #         for ep in epoch:
    #             dir = f'hd{hd}_nl{nl}_ep{ep}'
    #             log_file = dir + "/log.txt"
    #             save_path = dir + "/best_model.pt"
    #             loss_img = dir + "/loss.png"                                    

    #             experiments.append({
    #                 "input_dim": input_dim,
    #                 "hidden_dim": hd,
    #                 "out_dim": out_dim,
    #                 "num_layer": nl,
    #                 "activation": "relu",
    #                 "data_path": data_path,
    #                 "batch_size": batch_size,
    #                 "dropout": 0.1,
    #                 "device": device,
    #                 "optimizer": "sgd",
    #                 "lr": 0.01,
    #                 "epoch": ep,
    #                 "patience": patience,
    #                 "log_file": log_file,
    #                 "save_path": save_path,
    #                 "loss_img": loss_img
    #             })

    # 遍历激活函数     
    # for act in activation:
    #     dir = f'{act}'
    #     log_file = dir + "/log.txt"
    #     save_path = dir + "/best_model.pt"
    #     loss_img = dir + "/loss.png"
    #     experiments.append({
    #         "input_dim": input_dim,
    #         "hidden_dim": 128,
    #         "out_dim": out_dim,
    #         "num_layer": 4,
    #         "activation": act,
    #         "data_path": data_path,
    #         "batch_size": batch_size,
    #         "dropout": 0.1,
    #         "device": device,
    #         "optimizer": "sgd",
    #         "lr": 0.01,
    #         "epoch": 400,
    #         "patience": patience,
    #         "log_file": log_file,
    #         "save_path": save_path,
    #         "loss_img": loss_img
    #     })
    
    # 遍历 optimizer
    # for opt in optimizer: 
    #     dir = f'{opt}'
    #     log_file = dir + "/log.txt"
    #     save_path = dir + "/best_model.pt"
    #     loss_img = dir + "/loss.png"
    #     experiments.append({
    #         "input_dim": input_dim,
    #         "hidden_dim": 128,
    #         "out_dim": out_dim,
    #         "num_layer": 4,
    #         "activation": "relu",
    #         "data_path": data_path,
    #         "batch_size": batch_size,
    #         "dropout": 0.1,
    #         "device": device,
    #         "optimizer": opt,
    #         "lr": 0.01,
    #         "epoch": 400,
    #         "patience": patience,
    #         "log_file": log_file,
    #         "save_path": save_path,
    #         "loss_img": loss_img
    #     })
    
    # 遍历 lr
    # for learn_rate in lr:
    #     dir = f'learn_rate{learn_rate}'
    #     log_file = dir + "/log.txt"
    #     save_path = dir + "/best_model.pt"
    #     loss_img = dir + "/loss.png"
    #     experiments.append({
            
    #         "input_dim": input_dim,
    #         "hidden_dim": 128,
    #         "out_dim": out_dim,
    #         "num_layer": 4,
    #         "activation": "relu",
    #         "data_path": data_path,
    #         "batch_size": batch_size,
    #         "dropout": 0.1,
    #         "device": device,
    #         "optimizer": "sgd",
    #         "lr": learn_rate,
    #         "epoch": 400,
    #         "patience": patience,
    #         "log_file": log_file,
    #         "save_path": save_path,
    #         "loss_img": loss_img
    #     })
    
    # 重新遍历'sigmoid', 'softmax'     
    # activation = {'sigmoid', 'softmax'}
    # for act in activation:
    #     dir = f'{act}'
    #     log_file = dir + "/log.txt"
    #     save_path = dir + "/best_model.pt"
    #     loss_img = dir + "/loss.png"
    #     experiments.append({
    #         "input_dim": input_dim,
    #         "hidden_dim": 128,
    #         "out_dim": out_dim,
    #         "num_layer": 4,
    #         "activation": act,
    #         "data_path": data_path,
    #         "batch_size": batch_size,
    #         "dropout": 0.1,
    #         "device": device,
    #         "optimizer": "sgd",
    #         "lr": 0.01,
    #         "epoch": 400,
    #         "patience": 400, # 不早停
    #         "log_file": log_file,
    #         "save_path": save_path,
    #         "loss_img": loss_img
    #     })
    
    # # 遍历 dropout
    # for dr in dropout:
    #     dir = f'dropout{dr}'
    #     log_file = dir + "/log.txt"
    #     save_path = dir + "/best_model.pt"
    #     loss_img = dir + "/loss.png"
    #     experiments.append({
    #         "input_dim": input_dim,
    #         "hidden_dim": 128,
    #         "out_dim": out_dim,
    #         "num_layer": 4,
    #         "activation": "relu",
    #         "data_path": data_path,
    #         "batch_size": batch_size,
    #         "dropout": dr,
    #         "device": device,
    #         "optimizer": "sgd",
    #         "lr": 0.01,
    #         "epoch": 400,
    #         "patience": patience,
    #         "log_file": log_file,
    #         "save_path": save_path,
    #         "loss_img": loss_img
    #     })
    
    # 遍历 batch_size
    for bs in [32, 64, 128, 256]:
        dir = f'batch_size{bs}'
        log_file = dir + "/log.txt"
        save_path = dir + "/best_model.pt"
        loss_img = dir + "/loss.png"
        experiments.append({
            "input_dim": input_dim,
            "hidden_dim": 128,
            "out_dim": out_dim,
            "num_layer": 4,
            "activation": "sigmoid",
            "data_path": data_path,
            "batch_size": bs,
            "dropout": 0.1,
            "device": device,
            "optimizer": "sgd",
            "lr": 0.01,
            "epoch": 400,
            "patience": patience,
            "log_file": log_file,
            "save_path": save_path,
            "loss_img": loss_img
        })
    
    # 依次执行所有实验
    for exp in experiments:
        success, cmd_str = run_command(exp)
        if not success:
            with open('log.txt', 'a', encoding='utf-8') as f:
                f.write(f"❌ Failed Command:\n{cmd_str}\n\n")

    print("\n所有任务执行完毕！")