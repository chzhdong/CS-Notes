# Position Embedding

## RoPE

```python
import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1 / (base ** (torch.arange(0, dim, 2, dtype = torch.int64).float() / dim))
        self.register_buffer('inv_freq', tensor = inv_freq, persistent = False)

    torch.no_grad()
    def forward(self, x, position_ids):
        self.inv_freq.to(x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim = -1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 =x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim = -1)

def apply_emb(q, k, cos, sin, unsqueeze_dim = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# 测试代码
if __name__ == "__main__":
    # 设置随机种子以保证结果可重复
    torch.manual_seed(42)

    # 测试 RoPE 类
    dim = 4
    rope = RoPE(dim)
    x = torch.randn(2, 3, dim)  # 输入张量
    position_ids = torch.arange(3).unsqueeze(0).expand(2, -1)  # 位置索引
    cos, sin = rope(x, position_ids)
    print("cos:", cos)
    print("sin:", sin)

    # 测试 rotate_half 函数
    test_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rotated = rotate_half(test_input)
    print("rotate_half 输入:", test_input)
    print("rotate_half 输出:", rotated)

    # 测试 apply_emb 函数
    q = torch.randn(2, 3, dim)  # 查询向量
    k = torch.randn(2, 3, dim)  # 键向量
    q_rotated, k_rotated = apply_emb(q, k, cos, sin)
    print("apply_emb 后的 q:", q_rotated)
    print("apply_emb 后的 k:", k_rotated)
```

