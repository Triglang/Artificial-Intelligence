from torch import nn
import torch

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim,
        out_dim,
        device,
        num_layers,
        activation,
        dropout
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        
        # 参数初始化
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 激活函数映射表
        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1),
            'none': nn.Identity()
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activation_map[activation]

        # 动态构建网络层
        layers = []
        current_dim = input_dim
        
        """
        构建隐藏层（num_layers-1层）
        生成的网络结构为：

        Sequential(
        (0): Linear(in_features=784, out_features=128, bias=True)
        (1): activation()
        (2): Dropout
        (3): Linear(in_features=128, out_features=128, bias=True)
        (4): activation()
        (5): Dropout
        (6): Linear(in_features=128, out_features=10, bias=True)
        )
        """
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(current_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ]
            current_dim = hidden_dim  # 后续层输入维度统一
        
        # 添加输出层（无激活函数）
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        # 组合成顺序模型
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x):
        # 直接调用网络序列
        return self.net(x)

    # 设备管理
    def set_device(self, device: torch.device):
        self.to(device)
        self.device = device
        return self

    # 模型保存/加载
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))