#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils/tensor_description.h"

#define BLOCK_SIZE 32

__constant__ int c_width;
__constant__ int c_height;
__constant__ int c_channel_stride;

__global__ void CalculateEdgeWeightsKernel(const float* input, float* output) {
    int x = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    int y = threadIdx.y + blockIdx.y * BLOCK_SIZE;

    if (x >= c_width) return;
    if (y >= c_height) return;

    int home = x + y * c_width;
    float home_value = input[home];

    bool is_edge[4] = {x == c_width-1, y == c_height-1, x == 0, y == 0};
    int offsets[4] = {1, c_width, -1, -c_width};

    #pragma unroll
    for (int i = 0; i < 4; i++){
        float diff = is_edge[i] ? 0 : home_value - input[home + offsets[i]];
        output[home + i * c_channel_stride] = exp(1-(diff * diff));
    }
}

torch::Tensor CalculateEdgeWeights_Cuda(torch::Tensor input_tensor)
{
    TensorDescription desc = TensorDescription(input_tensor);

    torch::Tensor output_tensor = torch::zeros({desc.batchCount, 4, desc.sizes[0], desc.sizes[1]}, input_tensor.device());

    cudaMemcpyToSymbol(c_width, &desc.sizes[0], sizeof(int));
    cudaMemcpyToSymbol(c_height, &desc.sizes[1], sizeof(int));
    cudaMemcpyToSymbol(c_channel_stride, &desc.channelStride, sizeof(int));

    int block_count_x = int(desc.sizes[0] / BLOCK_SIZE) + 1;
    int block_count_y = int(desc.sizes[1] / BLOCK_SIZE) + 1;
    int block_count_z = desc.batchCount;

    dim3 block_count = dim3(block_count_x, block_count_y, block_count_z);
    dim3 block_size = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

    CalculateEdgeWeightsKernel<<<block_count, block_size>>>(input_tensor.data_ptr<float>(), output_tensor.data_ptr<float>());

    return output_tensor;
}
