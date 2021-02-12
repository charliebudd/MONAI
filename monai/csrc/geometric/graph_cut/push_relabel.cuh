#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_SIZE 32

__constant__ int c_width;
__constant__ int c_height;
__constant__ int c_element_count;
__constant__ int c_block_count_x;
__constant__ int c_iteration;

class PushRelabel
{
public:

    PushRelabel(torch::Tensor edge_weights_tensor, torch::Tensor source_weights_tensor, torch::Tensor sink_weights_tensor);
    ~PushRelabel();

    torch::Tensor Execute(int iterations, int a, int b);

private:

    void CompactActiveBlockMap();
    void SwapHeightBuffer();

private:

    int width, height;
    int total_block_count, active_block_count;
    int block_thread_count;

    int *active_block_map_host, *active_block_map;
    int *edge_capacities;
    int *excess_flow;
    int *height_read, *height_write, *height_swap;
};