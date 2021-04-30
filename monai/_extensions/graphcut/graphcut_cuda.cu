/*
Copyright 2020 - 2021 MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include "graphcut.h"

#define PI 3.14159265358979323846
#define TILE(SIZE, STRIDE) ((((SIZE) - 1)/(STRIDE)) + 1)

#if DIMENSION_COUNT == 1
    #define BLOCK_SIZE 128
    #define THREAD_COUNT BLOCK_SIZE
    __constant__ const int c_neighbour_offsets[2][1] = {
        { 1},
        {-1},
    };
#elif DIMENSION_COUNT == 2
    #define BLOCK_SIZE 16
    #define THREAD_COUNT (BLOCK_SIZE * BLOCK_SIZE)
    __constant__ const int c_neighbour_offsets[8][2] = {
        { 1,  0},
        {-1,  0},
        { 0,  1},
        { 0, -1},

        { 1,  1},
        { 1, -1},
        {-1,  1},
        {-1, -1},   
    };
#elif DIMENSION_COUNT == 3
    #define BLOCK_SIZE 4
    #define THREAD_COUNT (BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE)
    __constant__ const int c_neighbour_offsets[26][3] = {
        { 1,  0,  0},
        {-1,  0,  0},
        { 0,  1,  0},
        { 0, -1,  0},
        { 0,  0,  1},
        { 0,  0, -1},

        { 0,  1,  1},
        { 0,  1, -1},
        { 0, -1,  1},
        { 0, -1, -1},
        { 1,  1,  0},
        { 1, -1,  0},
        { 1,  0,  1},
        { 1,  0, -1},
        { 1,  1,  1},
        { 1,  1, -1},
        { 1, -1,  1},
        { 1, -1, -1},
        {-1,  1,  0},
        {-1, -1,  0},
        {-1,  0,  1},
        {-1,  0, -1},
        {-1,  1,  1},
        {-1,  1, -1},
        {-1, -1,  1},
        {-1, -1, -1},
    };
#endif

__constant__ uint c_batch_count;
__constant__ uint c_element_count;
__constant__ uint c_image_sizes[DIMENSION_COUNT];
__constant__ uint c_image_strides[DIMENSION_COUNT];

__device__ void index_to_coordinate(uint index, uint coordinate[DIMENSION_COUNT])
{
    uint remainder = index;

    for (uint dim = 0; dim < DIMENSION_COUNT; dim++)
    {
        uint stride = c_image_strides[dim];
        uint coord = remainder / stride;

        coordinate[dim] = coord;
        remainder -= coord * stride;
    }
}

__device__ uint coordinate_to_index(const uint coordinate[DIMENSION_COUNT])
{
    uint index = 0;

    for (uint dim = 0; dim < DIMENSION_COUNT; dim++)
    {
        index += coordinate[dim] * c_image_strides[dim];
    }

    return index;
}

__device__ bool get_neighbour_coordinate(const uint home_coordinate[DIMENSION_COUNT], uint neighbour_coordinate[DIMENSION_COUNT], const uint connection_index)
{
    for (int d = 0; d < DIMENSION_COUNT; d++)
    {
        uint coordinate = home_coordinate[d] + c_neighbour_offsets[connection_index][d];

        if (coordinate == 0xffffffff || coordinate == c_image_sizes[d])
        {
            return false;
        }

        neighbour_coordinate[d] = coordinate;
    }

    return true;
}

__global__ void ComputeConnectivity(const float* g_image, float* g_connectivities)
{
    uint home_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (home_index >= c_element_count)
    {
        return;
    }

    uint home_coord[DIMENSION_COUNT];
    uint neighbour_coord[DIMENSION_COUNT];

    index_to_coordinate(home_index, home_coord);

    float home[CHANNEL_COUNT];

    for (uint channel_index = 0; channel_index < CHANNEL_COUNT; channel_index++)
    {
        home[channel_index] = g_image[channel_index * c_element_count + home_index];
    }

    for (uint connection_index = 0; connection_index < CONNECTION_COUNT; connection_index++)
    {
        bool vaild_neighbour = get_neighbour_coordinate(home_coord, neighbour_coord, connection_index);

        if (vaild_neighbour)
        {
            uint neighbour_index = coordinate_to_index(neighbour_coord);

            float color_distance_squared = 0.0f;

            for (uint channel_index = 0; channel_index < CHANNEL_COUNT; channel_index++)
            {
                float home_value = home[channel_index];
                float neighbour_value = g_image[channel_index * c_element_count + neighbour_index];

                float diff = home_value - neighbour_value;

                color_distance_squared += diff * diff;
            }

            int connectivity = exp(-2 * PI * color_distance_squared / CHANNEL_COUNT) * 255;

            g_connectivities[connection_index * c_element_count + home_index] = connectivity;
        }
        else
        {
            g_connectivities[connection_index * c_element_count + home_index] = -1;
        }
    }
}

void graphcut_cuda(const float* image, const float* weights, float* output, const uint batch_count, const uint element_count, const uint sizes[DIMENSION_COUNT], const uint strides[DIMENSION_COUNT])
{
    cudaMemcpyToSymbol(c_batch_count, &batch_count, sizeof(uint));
    cudaMemcpyToSymbol(c_element_count, &element_count, sizeof(uint));
    cudaMemcpyToSymbol(c_image_sizes, sizes, DIMENSION_COUNT * sizeof(uint));
    cudaMemcpyToSymbol(c_image_strides, strides, DIMENSION_COUNT * sizeof(uint));

    dim3 grid(TILE(element_count, THREAD_COUNT), 1, 1);
    dim3 block(THREAD_COUNT, 1, 1);

    ComputeConnectivity<<<grid, block>>>(image, output);
}
