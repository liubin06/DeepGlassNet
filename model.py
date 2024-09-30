import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepGlass(nn.Module):
    def __init__(self, input_dim=19, hidden_dim1=64, hidden_dim2=32, output_dim=16):
        super(DeepGlass, self).__init__()

        # 交叉特征提取层: 是两层线性变换层，分别用于将输入特征映射到隐层。首先将输入的19维特征映射到128维，然后再进一步映射到64维。这两层可以通过简单的线性操作隐式地学习输入特征之间的交互关系。
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)

        # 自注意力层
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim2, num_heads=1)  # 注意力模块
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim2, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, output_dim)

        # 批归一化和Dropout
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 线性变换层
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))

        # # 自注意力机制
        # x = x.unsqueeze(0)  # 增加一个维度，变成 [1, batch_size, hidden_dim2]
        # attn_output, _ = self.attention(x, x, x)
        # x = attn_output.squeeze(0)  # 恢复维度，变成 [batch_size, hidden_dim2]

        # 全连接层1
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        # 全连接层2
        x = self.fc2(x)
        l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)

        return x/l2_norm


# # # 示例：创建模型并进行前向传播
# model = DeepGlass()
# sample_input = torch.rand(32, 19)  # 32个样本，每个样本19个特征
# print(sample_input.shape)
# output = model(sample_input)
# print(output.shape)  # 输出应为 [32, 1]
#
