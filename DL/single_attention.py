import torch
import torch.nn as nn
import torch.nn.functional as F


# single_attention
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, k_dim,  v_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        
        self.to_Q = nn.Linear(self.embed_dim, self.k_dim)
        self.to_K = nn.Linear(self.embed_dim, self.k_dim)
        self.to_V = nn.Linear(self.embed_dim, self.v_dim)

    def forward(self ,x ):
        batch_size, seq_length, embed_dim = x.shape

        Q = self.to_Q(x)
        K = self.to_K(x)
        V = self.to_V(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))/ torch.sqrt(torch.tensor(self.k_dim, dtype=torch.float32))
        attention = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention, V)
        return out


if __name__ == "__main__":
    # 参数设置
    embed_dim = 16  # 输入特征维度
    k_dim = 8       # Q 和 K 的特征维度
    v_dim = 10      # V 的特征维度
    batch_size = 2  # 批量大小
    seq_length = 5  # 序列长度

    # 创建自注意力模型
    self_attention = SelfAttention(embed_dim, k_dim, v_dim)

    # 创建随机输入张量
    x = torch.randn(batch_size, seq_length, embed_dim)  # 输入形状 (batch_size, seq_length, embed_dim)

    # 前向传播
    output = self_attention(x)

    # 验证输出形状
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, seq_length, v_dim), "Output shape is incorrect!"

    print("Test passed! SelfAttention output is correct.")