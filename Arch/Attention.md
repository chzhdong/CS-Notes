# Attention

## SelfAttention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):

        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, masked = None):
        batch_size = x.shape[0]

        Q = w_q(x)
        K = w_k(x)
        V = w_v(x)

        scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32), dim = -1)

        if masked is not None:
            scores = scores.masked_fill(mask == 0, 1e-10)

        output = F.softmax(scores, dim = -1)
        output = self.dropout(output)

        return torch.matmul(scores, V)
```

## MultiHeadAttention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask = None):
        batch_size = x.shape[0]

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32), dim = -1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        return self.w_o(output)

if __name__ == "__main__":
    batch_size = 8
    seq_len = 16
    d_model = 64
    num_heads = 8

    x = torch.randn(batch_size, seq_len, d_model)

    model = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

    output = model(x)

    print(f"o: {output.shape}")
```


## GroupQueryAttention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, group_num):
        super().__init__()

        self.d_model = d_model
        self.group_num = group_num

        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model // self.group_num)
        self.w_v = nn.Linear(self.d_model, self.d_model // self.group_num)
        self.w_o = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, masked = None):
        batch_size = x.shape[0]

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q_groups = torch.chunk(Q, self.group_num, -1)
        attn_scores = []
        for q in Q_groups:
            scores = q @ K.T / torch.sqrt(torch,tensor(d_model // self.group_num, dtype = torch.float32), dim = -1)
            scores = F.softmax(scores, dim = -1)
            attn_scores.append(scores)
        
        outputs = []
        for score in attn_scores:
            output = score @ V
            outputs.append(output)

        return self.w_o(torch.cat([output for output in outputs], dim = -1))
```

## CrossAttention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_model):
        super.__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x1, x2):
        Q = self.w_q(x1)
        K = self.w_k(x2)
        V = self.w_v(x2)

        scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(self.d_model, dtype = torch.float32), dim = -1)

        scores = F.softmax(scores, dim = -1)

        return torch.matmul(scores, V)
```