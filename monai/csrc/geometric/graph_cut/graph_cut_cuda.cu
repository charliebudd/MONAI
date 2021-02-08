#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include "utils/tensor_description.h"

#include <iostream>

#define BLOCK_SIZE 32
#define HEIGHT_MAX 15
#define ERR_CHK //py::print(cudaGetErrorString(cudaDeviceSynchronize()));

__constant__ int c_width;
__constant__ int c_height;
__constant__ int c_element_count;
__constant__ int c_block_count_x;
__constant__ int c_iteration;

//###########################################
// Initialises the excess fow and the active block map
__global__ void InitialisationKernel(const float* edge_weights, const float* source_weights, const float* sink_weights, int* edge_capacities, int* excess_flow, int* active_block_map) 
{    
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int block_y = block_id / c_block_count_x;
    int block_x = block_id - block_y * c_block_count_x;

    int thread_y = thread_id / BLOCK_SIZE;
    int thread_x = thread_id - thread_y * BLOCK_SIZE;

    int x = thread_x + block_x * BLOCK_SIZE;
    int y = thread_y + block_y * BLOCK_SIZE;

    if (x >= c_width) return;
    if (y >= c_height) return;

    int home = x + y * c_width;

    // writing edge capacities
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        edge_capacities[home + i * c_element_count] = edge_weights[home + i * c_element_count] * HEIGHT_MAX;
    }

    // initialising excess flow
    excess_flow[home] = (sink_weights[home] - source_weights[home]) * c_element_count;

    // initialising active block map
    if(thread_id == 0)
    {
        active_block_map[block_id] = block_id;
    }
}

//###########################################
// Disperses excess flow through the network based on edge capacities.
__global__ void PushKernel(const int* active_block_map, const int* height, int* edge_capacities, int* excess_flow) 
{    
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    int active_block = active_block_map[block_id];

    int block_y = active_block / c_block_count_x;
    int block_x = active_block - block_y * c_block_count_x;

    int thread_y = thread_id / BLOCK_SIZE;
    int thread_x = thread_id - thread_y * BLOCK_SIZE;

    int x = thread_x + block_x * BLOCK_SIZE;
    int y = thread_y + block_y * BLOCK_SIZE;

    if (x >= c_width) return;
    if (y >= c_height) return;

    int home = x + y * c_width;
    int home_height = height[home];
    int home_flow = excess_flow[home];

    if (home_height >= HEIGHT_MAX || home_flow <= 0) return;
    
    bool is_edge[4] = {x == c_width-1, y == c_height-1, x == 0, y == 0};
    int offsets[4] = {1, c_width, -1, -c_width};
    int redirect[4] = {2, 3, 0, 1};
    
    #pragma unroll
    for (int i = 0; i < 4 && home_flow > 0; i++) 
    {
        int neighbour = home + offsets[i];

        if (!is_edge[i] && height[neighbour] == home_height - 1) 
        {
            int home_to_neighbour = home + i * c_element_count;
            int neighbour_to_home = neighbour + redirect[i] * c_element_count;

            int home_edge = edge_capacities[home_to_neighbour];
            float edge_flow = min(home_edge, home_flow);
            
            if(edge_flow != 0)
            {
                atomicSub(edge_capacities + home_to_neighbour, edge_flow);
                atomicAdd(edge_capacities + neighbour_to_home, edge_flow);
    
                atomicSub(excess_flow + home, edge_flow);
                atomicAdd(excess_flow + neighbour, edge_flow);

                home_flow -= edge_flow;
            }
        }
    }
}

//###########################################
// Sets the height of each node based on excess flow 
__global__ void RelabelKernel(const int* active_block_map, const int* height_read, const int* edge_capacities, const int* excess_flow, int* height_write) 
{
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    int active_block = active_block_map[block_id];

    int block_y = active_block / c_block_count_x;
    int block_x = active_block - block_y * c_block_count_x;

    int thread_y = thread_id / BLOCK_SIZE;
    int thread_x = thread_id - thread_y * BLOCK_SIZE;

    int x = thread_x + block_x * BLOCK_SIZE;
    int y = thread_y + block_y * BLOCK_SIZE;

    if (x >= c_width) return;
    if (y >= c_height) return;

    int home = x + y * c_width;
    int home_height = height_read[home];
    int home_flow = excess_flow[home];

    if (home_height >= HEIGHT_MAX || home_flow <= 0) 
    { 
        height_write[home] = home_height; 
        return; 
    }

    bool is_edge[4] = {x == c_width-1, y == c_height-1, x == 0, y == 0};
    int offsets[4] = {1, c_width, -1, -c_width};

    int min_height = HEIGHT_MAX;

    #pragma unroll
    for (int i = 0; i < 4; i++) 
    {
        if (!is_edge[i] && edge_capacities[home + i * c_element_count] > 0)
        {
            int neighbour_height = height_read[home + offsets[i]];
            min_height = min(min_height, neighbour_height + 1);
        }
    }

    height_write[home] = min_height;
}

//###########################################
// Checks for any active nodes within block
__global__ void ActiveBlockCheckKernel(const int* height, const int* excess_flow, int* active_block_map) 
{
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    int block_y = block_id / c_block_count_x;
    int block_x = block_id - block_y * c_block_count_x;

    int thread_y = thread_id / BLOCK_SIZE;
    int thread_x = thread_id - thread_y * BLOCK_SIZE;

    int x = thread_x + block_x * BLOCK_SIZE;
    int y = thread_y + block_y * BLOCK_SIZE;

    if (x >= c_width) return;
    if (y >= c_height) return;

    int home = x + y * c_width;
    int home_height = height[home];
    int home_flow = excess_flow[home];

    bool active = home_height < HEIGHT_MAX && home_flow > 0;
    bool any_active = __syncthreads_or(active);

    if(threadIdx.x == 0)
    {
        active_block_map[block_id] = any_active ? block_id : -1;
    } 
}

//###########################################
// Checks for any active nodes within block
__global__ void GlobalRelabelKernel(const int* excess_flow, int* height_read, int* height_write) 
{
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    int block_y = block_id / c_block_count_x;
    int block_x = block_id - block_y * c_block_count_x;

    int thread_y = thread_id / BLOCK_SIZE;
    int thread_x = thread_id - thread_y * BLOCK_SIZE;

    int x = thread_x + block_x * BLOCK_SIZE;
    int y = thread_y + block_y * BLOCK_SIZE;

    if (x >= c_width) return;
    if (y >= c_height) return;

    int home = x + y * c_width;

    if (c_iteration == 0)
    {
        height_write[home] = excess_flow[home] < 0 ? 0 : -1;
        return;
    }
    else if (height_read[home] == -1)
    {
        bool is_edge[4] = {x == c_width-1, y == c_height-1, x == 0, y == 0};
        int offsets[4] = {1, c_width, -1, -c_width};

        bool relabel = false;

        #pragma unroll
        for (int i = 0; i < 4; i++) 
        {
            if (!is_edge[i])
            {
                int neighbour_height = height_read[home + offsets[i]];
                relabel |= neighbour_height == (c_iteration - 1);
            }
        }

        if (relabel)
        {
            height_write[home] = c_iteration;
        }
    }
}

//###########################################

__global__ void TEMP_OUTPUT_DUMP(const int* height, const int* excess_flow, const int* edge_capacities, int* output) 
{
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    int block_y = block_id / c_block_count_x;
    int block_x = block_id - block_y * c_block_count_x;

    int thread_y = thread_id / BLOCK_SIZE;
    int thread_x = thread_id - thread_y * BLOCK_SIZE;

    int x = thread_x + block_x * BLOCK_SIZE;
    int y = thread_y + block_y * BLOCK_SIZE;

    if (x >= c_width) return;
    if (y >= c_height) return;

    int home = x + y * c_width;
    int home_height = height[home];
    int home_flow = excess_flow[home];
    
    bool active = home_height < HEIGHT_MAX && home_flow > 0;
    int any_active = __syncthreads_or(active);

    output[home + 0 * c_element_count] = home_flow;
    output[home + 1 * c_element_count] = home_height;
    output[home + 2 * c_element_count] = edge_capacities[home + 0 * c_element_count];
    output[home + 3 * c_element_count] = edge_capacities[home + 1 * c_element_count];
    output[home + 4 * c_element_count] = edge_capacities[home + 2 * c_element_count];
    output[home + 5 * c_element_count] = edge_capacities[home + 3 * c_element_count];
    output[home + 6 * c_element_count] = any_active;
    output[home + 7 * c_element_count] = active;
}

//###########################################

torch::Tensor GraphCut_Cuda(torch::Tensor edge_weights_tensor, torch::Tensor source_weights_tensor, torch::Tensor sink_weights_tensor, int iterations)
{
    TensorDescription desc = TensorDescription(edge_weights_tensor);

    float *edge_weights = edge_weights_tensor.data_ptr<float>();
    float *source_weights = source_weights_tensor.data_ptr<float>();
    float *sink_weights = sink_weights_tensor.data_ptr<float>();

    int dimensions = desc.dimensions;
    int width = desc.sizes[0];
    int height = desc.sizes[1];
    int element_count = width * height;

    int block_count_x = int(width / BLOCK_SIZE) + 1;
    int block_count_y = int(height / BLOCK_SIZE) + 1;
    int block_count = block_count_x * block_count_y;

    int block_thread_count = pow(BLOCK_SIZE, dimensions);

    int *active_block_map_host, *active_block_map;
    active_block_map_host = new int[block_count];
    cudaMalloc(&active_block_map, block_count * sizeof(int));

    int *edge_capacities;
    cudaMalloc(&edge_capacities, 4 * element_count * sizeof(int));

    int *excess_flow;
    cudaMalloc(&excess_flow, element_count * sizeof(int));

    int *height_read, *height_write, *height_swap;
    cudaMalloc(&height_read, element_count * sizeof(int));
    cudaMalloc(&height_write, element_count * sizeof(int));
    
    cudaMemcpyToSymbol(c_width, &width, sizeof(int));
    cudaMemcpyToSymbol(c_height, &height, sizeof(int));
    cudaMemcpyToSymbol(c_block_count_x, &block_count_x, sizeof(int));
    cudaMemcpyToSymbol(c_element_count, &element_count, sizeof(int));

    int active_block_count = block_count;

    // Initialisation...
    InitialisationKernel<<<active_block_count, block_thread_count>>>(edge_weights, source_weights, sink_weights, edge_capacities, excess_flow, active_block_map);

    int iteration_counter = 0;

    int stuck_counter = 0;
    int stuck_limit = 100;

    // Iterating while any blocks are active...
    while (active_block_count > 0)
    {
        if(iteration_counter >= iterations)
        {
            //py::print("Iteration Count Reached. Escaping...");
            break;
        }
        
        iteration_counter++;

        //py::print("active block count: ", active_block_count);

        RelabelKernel<<<active_block_count, block_thread_count>>>(active_block_map, height_read, edge_capacities, excess_flow, height_write);
        height_swap = height_read; 
        height_read = height_write; 
        height_write = height_swap;
        ERR_CHK

        PushKernel<<<active_block_count, block_thread_count>>>(active_block_map, height_read, edge_capacities, excess_flow);
        ERR_CHK

        ActiveBlockCheckKernel<<<block_count, block_thread_count>>>(height_read, excess_flow, active_block_map);
        ERR_CHK

        //######################################
        // host side stream compaction for now

        cudaMemcpy(active_block_map_host, active_block_map, block_count * sizeof(int), cudaMemcpyDeviceToHost);
        ERR_CHK

        int previous_active_block_count = active_block_count;
        active_block_count = 0;

        //py::print("block activations...");
        for (int i = 0; i < block_count; i++)
        {
            //py::print(i, ": ", active_block_map_host[i] != -1);

            if(active_block_map_host[i] != -1) 
            {
                active_block_map_host[active_block_count] = active_block_map_host[i];
                active_block_count++;
            }
        }

        cudaMemcpy(active_block_map, active_block_map_host, active_block_count * sizeof(int), cudaMemcpyHostToDevice);
        ERR_CHK
        
        //######################################

        if (active_block_count == previous_active_block_count)
        {
            stuck_counter++;

            if (stuck_counter > stuck_limit)
            {
                py::print("Sticking Limit Reached. Escaping...");
                break;
            }
        }
        else
        {
            stuck_counter = 0;
        }
    }

    // for (int i = 0; i < 100; i++)
    // {
    //     cudaMemcpyToSymbol(c_iteration, &i, sizeof(int));
    //     GlobalRelabelKernel<<<block_count, block_thread_count>>>(excess_flow, height_read, height_write);
    //     height_swap = height_read; 
    //     height_read = height_write; 
    //     height_write = height_swap;
    // }

    //py::print("done");
    ERR_CHK

    torch::Tensor output_tensor = torch::zeros({1, 8, width, height}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0));

    TEMP_OUTPUT_DUMP<<<block_count, block_thread_count>>>(height_read, excess_flow, edge_capacities, output_tensor.data_ptr<int>());

    return output_tensor;
}
