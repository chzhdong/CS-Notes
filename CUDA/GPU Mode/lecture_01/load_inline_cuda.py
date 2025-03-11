import os
import torch
from torch.utils.cpp_extension import load_inline

os.environ['TORCH_CUDA_ARCH_LIST'] = 'Volta'

cuda_source = """
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < height) {
        int index = row * height + col;
        result[index] = matrix[index] * matrix[index];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const int height = matrix.size(0);
    const int width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x, 
                         (height + threads_per_block.y - 1) / threads_per_block.y);
    
    square_matrix_kernel<<<blocks_per_grid, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);
    return result;
}
"""

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

square_matrix_extension = load_inline(
    name = "square_matrix_extension",
    cpp_sources = cpp_source,
    cuda_sources = cuda_source,
    functions = ['square_matrix'],
    with_cuda = True,
    extra_cuda_cflags = ["-O2"],
    build_directory = "./square_matrix"
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device = 'cuda')
print(square_matrix_extension.square_matrix(a))
