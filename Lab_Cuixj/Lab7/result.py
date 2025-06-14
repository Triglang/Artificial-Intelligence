import matplotlib.pyplot as plt

def hd_nl_ep_result(epoch, hidden_dim, num_layers):
    para = {"accuracy": None, "average_f1": None}
    if hidden_dim == 32:
        if num_layers == 2:
            if epoch == 300:
                para["accuracy"] = 0.9496
                para["average_f1"] = 0.9491
            elif epoch == 400:
                para["accuracy"] = 0.9553
                para["average_f1"] = 0.9548  
            elif epoch == 500:
                para["accuracy"] = 0.9577
                para["average_f1"] = 0.9572
            else:
                print("epoch error")
        elif num_layers == 3:
            if epoch == 300:
                para["accuracy"] = 0.9536
                para["average_f1"] = 0.9531
            elif epoch == 400:
                para["accuracy"] = 0.9590
                para["average_f1"] = 0.9585
            elif epoch == 500:
                para["accuracy"] = 0.9629
                para["average_f1"] = 0.9625
            else:
                print("epoch error")
        elif num_layers == 4:
            if epoch == 300:
                para["accuracy"] = 0.9538
                para["average_f1"] = 0.9532
            elif epoch == 400:
                para["accuracy"] = 0.9620
                para["average_f1"] = 0.9617 
            elif epoch == 500:
                para["accuracy"] = 0.9599
                para["average_f1"] = 0.9592 
            else:
                print("epoch error")
        else:
            print("num_layers error")
    elif hidden_dim == 64:
        if num_layers == 2:
            if epoch == 300:
                para["accuracy"] = 0.9545
                para["average_f1"] = 0.9541
            elif epoch == 400:
                para["accuracy"] = 0.9611
                para["average_f1"] = 0.9608
            elif epoch == 500:
                para["accuracy"] = 0.9658
                para["average_f1"] = 0.9655
            else:
                print("epoch error")
        elif num_layers == 3:
            if epoch == 300:
                para["accuracy"] = 0.9662
                para["average_f1"] = 0.9660
            elif epoch == 400:
                para["accuracy"] = 0.9680
                para["average_f1"] = 0.9678
            elif epoch == 500:
                para["accuracy"] = 0.9709
                para["average_f1"] = 0.9707 
            else:
                print("epoch error")
        elif num_layers == 4:
            if epoch == 300:
                para["accuracy"] = 0.9697
                para["average_f1"] = 0.9694
            elif epoch == 400:
                para["accuracy"] = 0.9707
                para["average_f1"] = 0.9705
            elif epoch == 500:
                para["accuracy"] = 0.9719
                para["average_f1"] = 0.9717
            else:
                print("epoch error")
        else:
            print("num_layers error")
    elif hidden_dim == 128:
        if num_layers == 2:
            if epoch == 300:
                para["accuracy"] = 0.9570
                para["average_f1"] = 0.9566
            elif epoch == 400:
                para["accuracy"] = 0.9647
                para["average_f1"] = 0.9644
            elif epoch == 500:
                para["accuracy"] = 0.9677
                para["average_f1"] = 0.9675
            else:
                print("epoch error")
        elif num_layers == 3:
            if epoch == 300:
                para["accuracy"] = 0.9696
                para["average_f1"] = 0.9694
            elif epoch == 400:
                para["accuracy"] = 0.9724
                para["average_f1"] = 0.9722
            elif epoch == 500:
                para["accuracy"] = 0.9757
                para["average_f1"] = 0.9756
            else:
                print("epoch error")
        elif num_layers == 4:
            if epoch == 300:
                para["accuracy"] = 0.9726
                para["average_f1"] = 0.9723
            elif epoch == 400:
                para["accuracy"] = 0.9754
                para["average_f1"] = 0.9752
            elif epoch == 500:
                para["accuracy"] = 0.9780
                para["average_f1"] = 0.9778
            else:
                print("epoch error")
        else:
            print("num_layers error")
    else:
        print("hidden_dim error")
    
    return para

def plot_hd_nl_ep():
    hidden_dim_set = [32, 64, 128]
    num_layers_set = [2, 3, 4]
    epoch_set = [300, 400, 500]
    
    bar_width = 0.2
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 颜色对应层数
    
    for epoch in epoch_set:
        plt.figure(figsize=(12, 6))
        all_bars = []  # 记录所有柱子对象
        
        # 遍历组合绘制柱子
        for hd_idx, hidden_dim in enumerate(hidden_dim_set):
            for nl_idx, num_layers in enumerate(num_layers_set):
                data = hd_nl_ep_result(epoch, hidden_dim, num_layers)
                if not data['accuracy']:
                    continue
                
                # 计算坐标和值
                y_value = (data['accuracy'] - 0.95) * 10000
                x_pos = hd_idx + (nl_idx - 1) * bar_width
                
                # 绘制并记录柱子
                bar = plt.bar(
                    x_pos, 
                    y_value, 
                    width=bar_width,
                    color=colors[nl_idx],
                    label=f'{num_layers} Layers' if hd_idx == 0 else ""
                )
                all_bars.append(bar[0])  # 保存Bar对象

        # 添加数值标签
        for bar in all_bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,  # X坐标居中
                height + 0.5,                      # Y坐标微调偏移
                f'{height:.1f}',                   # 保留1位小数
                ha='center',                       # 水平居中
                va='bottom',                       # 垂直底部对齐
                fontsize=8,
                color='black'
            )
        
        # 设置图表属性
        plt.title(f'Accuracy vs Hidden Dimension (Epoch {epoch})', fontsize=14)
        plt.xlabel('Hidden Dimension', fontsize=12)
        plt.ylabel('(Accuracy - 0.95) × 10000', fontsize=12)
        plt.xticks(
            ticks=range(len(hidden_dim_set)),
            labels=[str(hd) for hd in hidden_dim_set],
            fontsize=10
        )
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 处理图例
        handles, labels = plt.gca().get_legend_handles_labels()
        legend_labels = dict(zip(labels, handles))  # 去重
        plt.legend(
            legend_labels.values(), 
            legend_labels.keys(), 
            title='Number of Layers',
            loc='upper left'
        )
        
        # 调整坐标轴范围
        plt.xlim(-0.5, len(hidden_dim_set)-0.5)
        
        plt.tight_layout()
        plt.savefig(f'accuracy_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

def opt_result(opt):
    para = {"accuracy": None, "average_f1": None}
    if opt == "sgd":
        para["accuracy"] = 0.9756
        para["average_f1"] = 0.9753
    elif opt == "adam":
        para["accuracy"] = 0.9774
        para["average_f1"] = 0.9772
    elif opt == "adamw":
        para["accuracy"] = 0.9751
        para["average_f1"] = 0.9749
    else:
        print("optimizer error")
    
    return para

def plot_opt():
    optimizers = ["sgd", "adam", "adamw"]
    y_values = []
    
    # 获取数据
    for opt in optimizers:
        data = opt_result(opt)
        y_values.append((data["accuracy"] - 0.95) * 10000)

    # 创建图表
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        optimizers, 
        y_values,
        width=0.4,
        color=['#1f77b4', '#2ca02c', '#d62728']  # 蓝/绿/红
    )
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.5,  # 数值标签位置偏移
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # 设置图表属性
    plt.title("Optimizer Performance Comparison", pad=20)
    plt.xlabel("Optimizer Type", labelpad=10)
    plt.ylabel("(Accuracy - 0.95) × 10000", labelpad=10)
    plt.xticks(fontsize=9)
    
    # 设置纵轴范围
    plt.ylim(240, 280)
    
    # 保存并关闭
    plt.savefig('optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()        

def lr_result(learn_rate):
    para = {"accuracy": None, "average_f1": None}
    if learn_rate == 0.001:
        para["accuracy"] = 0.8976
        para["average_f1"] = 0.8954
    elif learn_rate == 0.01:
        para["accuracy"] = 0.9763
        para["average_f1"] = 0.9761
    elif learn_rate == 1:
        para["accuracy"] = 0.9754
        para["average_f1"] = 0.9752
    else:
        print("Error: Unsupported learning rate")
    return para  # 确保返回数据

def plot_lr():
    lr_set = [0.001, 0.01, 1]
    lr_labels = [f"{lr:.3f}" if lr < 1 else "1.000" for lr in lr_set]  # 统一显示格式
    accuracies = []
    
    # 获取准确率数据
    for lr in lr_set:
        data = lr_result(lr)
        accuracies.append(data["accuracy"])
    
    # 创建图表
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        lr_labels, 
        accuracies,
        width=0.4,
        color=['#2ca02c', '#1f77b4', '#d62728'],  # 不同学习率用不同颜色
        edgecolor='black'  # 添加黑色边框
    )
    
    # 添加精确数值标签
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        va_type = 'bottom' if idx != 0 else 'top'  # 最低柱子标签朝上
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + (0.005 if idx !=0 else -0.005),  # 避免标签重叠
            f'{height:.4f}',
            ha='center',
            va=va_type,
            fontsize=9,
            fontweight='bold'
        )
    
    # 图表样式设置
    plt.title("Learning Rate Impact on Model Accuracy", pad=15)
    plt.xlabel("Learning Rate", labelpad=10)
    plt.ylabel("Validation Accuracy", labelpad=10)
    plt.ylim(0.85, 0.98)  # 聚焦有效数据区间
    
    # 保存输出
    plt.savefig('lr_impact.png', dpi=400, bbox_inches='tight')
    plt.close()
    
def activation_result(activation):
    para = {"accuracy": None, "average_f1": None}
    if activation == "relu":
        para["accuracy"] = 0.9748
        para["average_f1"] = 0.9746
    elif activation == "tanh":
        para["accuracy"] = 0.9644
        para["average_f1"] = 0.9640
    elif activation == "sigmoid":
        para["accuracy"] = 0.3366
        para["average_f1"] = 0.2529
    elif activation == "softmax":
        para["accuracy"] = 0.1135
        para["average_f1"] = 0.0204
    else:
        print("Error: Unsupported activation function")
    return para  # 确保返回数据

def plot_activation():
    activations = ['relu', 'tanh', 'sigmoid', 'softmax']
    accuracies = []
    
    # 获取准确率数据
    for act in activations:
        data = activation_result(act)
        accuracies.append(data["accuracy"])
    
    # 创建可视化图表
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        activations,
        accuracies,
        width=0.4,
        color=['#2ca02c', '#1f77b4', '#d62728', '#ff7f0e'],  # 绿/蓝/红/橙
        edgecolor='black',  # 黑色边框
        alpha=0.85
    )
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        # 动态调整标签位置
        va_type = 'bottom' if height > 0.3 else 'top'
        y_pos = height + 0.02 if height > 0.3 else height - 0.02
        color = 'black' if height > 0.3 else 'white'
        
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            y_pos,
            f'{height:.4f}',
            ha='center',
            va=va_type,
            color=color,
            fontsize=10,
            fontweight='bold'
        )
    
    # 设置图表样式
    plt.title("Activation Function Performance Comparison", pad=20, fontsize=14)
    plt.xlabel("Activation Functions", labelpad=12, fontsize=12)
    plt.ylabel("Validation Accuracy", labelpad=12, fontsize=12)
    plt.ylim(0, 1.05)  # 设置统一坐标范围
    plt.xticks(fontsize=11)
    
    # 添加辅助网格线
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # 保存并关闭
    plt.savefig('activation_accuracy.png', dpi=400, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # plot_hd_nl_ep()
    # plot_opt()
    # plot_lr()
    plot_activation()