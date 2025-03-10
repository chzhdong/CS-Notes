# Flash Attention

 Flash Attention 前向传播 CUDA 源码及 Pytorch 绑定整理！

```cuda
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float* l,
    float* m,
    float* O) {
        // Get the thread and block index
        int tx = threadIdx.x;
        int bx = blockIdx.x;
        int by = blockIdx.y;

        // Offset into Q, K, V, l, m - different for each batch and block
        int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
        int lm_offset = (bx * gridDim.y * N) + (by * N);

        // Define SRAM for Q, K, V, S
        extern __shared__ float sram[];
        int tile_size = Bc * d; // Size of Qi, Kj, Vj
        float* Qi = sram;
        float* Kj = &sram[tile_size];
        float* Vj = &sram[2 * tile_size];
        float* S = &sram[3 * tile_size];

        for (int j = 0; j < Tc; j++) {
            // Load Kj, Vj to SRAM
            for (int x= 0; x < d; x++) {
                Kj[tx * d + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
                Vj[tx * d + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            }
            __syncthreads(); // Wait for all threads to load Kj, Vj
            for (int i = 0; i < Tr; i++) {
                // Load Qi to SRAM, l and m to registers
                for (int x = 0; x < d; x++) {
                    Qi[tx * d + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
                }
                float row_m_prev = m[lm_offset + (Br * i) + tx];
                float row_l_prev = l[lm_offset + (Br * i) + tx];

                // S = QK^T, row_m = row_max(S)
                float row_m = -INFINITY;
                for (int y = 0; y < Bc; y++) {
                    float sum = 0;
                    for (int x = 0; x < d; x++) {
                        sum += Qi[tx * d + x] * Kj[y * d + x];
                    }
                    sum *= softmax_scale;
                    S[Bc * tx + y] = sum;

                    if (sum > row_m) {
                        row_m = sum;
                    }
                }

                // P = exp(S = row_m), row_l = row_sum(P)
                float row_l = 0;
                for (int y = 0; y < Bc; y++) {
                    S[Bc * tx + y] = __expf(S[Bc * tx + y] - row_m);
                    row_l += S[Bc * tx + y];
                }

                // Compute new m and l
                float row_m_new = max(row_m, row_m_prev);
                float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

                // Write O, l, and m to HBM
                for (int x = 0; x < d; x++) {
                    float pv = 0;
                    for (int y = 0; y < Bc; y++) {
                        pv += S[Bc * tx + y] * Vj[y * d + x];
                    }
                    O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
                }
                m[lm_offset + (Br * i) + tx] = row_m_new;
                l[lm_offset + (Br * i) + tx] = row_l_new;
            }
            __syncthreads(); // Wait for all threads to finish processing i
        }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 32;
    const int Br = 32;

    const int B = Q.size(0);
    const int nh = Q.size(1);
    const int N = Q.size(2);
    const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc);
    const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize l, m, O to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    // Calculate SRAM size needed each block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        N,
        d,
        Tc,
        Tr,
        Bc,
        Br,
        softmax_scale,
        l.data_ptr<float>(),
        m.data_ptr<float>(),
        O.data_ptr<float>()
    );
    return O;
}
```



```cpp
#include <torch/extension.h>

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", torch::wrap_pybind_function(forward), "forward");
}
```



```python
import os
import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

os.environ['TORCH_CUDA_ARCH_LIST'] = 'Volta'

# Load the CUDA kernel as a python module
flash_attn = load(name="flash_attn", sources=["main.cpp", "flash_attn.cu"], extra_cuda_cflags=['-O2'])

# Use small model parameters
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

# flash attention aims to faster
def manual_attn(q, k, v):
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
    attn = F.softmax(attn, dim=-1)
    y = attn @ v
    return y

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling cuda flash attention === ')

with torch.autograd.profiler.profile(use_device = 'cuda') as prof:
    cuda_result = flash_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(cuda_result, manual_result, rtol=0, atol=1e-02))
```

