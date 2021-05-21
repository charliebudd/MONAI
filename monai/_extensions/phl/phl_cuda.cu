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

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "phl.h"

#define TILE(SIZE, STRIDE) ((((SIZE) - 1)/(STRIDE)) + 1)

struct LatticeEntry
{
    int element_index;
    int lattice_coordinate[ELEVATED_COUNT - 1];
    float barycentric_weight;

    bool operator ==(const LatticeEntry &other)
    {
        bool result = true;

        for (int i = 0; i < ELEVATED_COUNT - 1; i++)
        {
            int a = lattice_coordinate[i];
            int b = other.lattice_coordinate[i];

            result &= a == b;
        }

        return result;
    }

    bool operator <(const LatticeEntry &other)
    {
        for (int i = 0; i < ELEVATED_COUNT - 1; i++)
        {
            int a = lattice_coordinate[i];
            int b = other.lattice_coordinate[i];

            if (a > b)
            {
                return false;
            }

            if (a < b)
            {
                return true;
            }
        }

        return true;
    }
};

__global__ void create_lattice(const float* g_features, LatticeEntry* g_lattice, const uint element_count)
{
    uint global_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint batch_index = blockIdx.y;

    const float* g_batch_features = g_features + batch_index * FEATURE_COUNT * element_count;
    LatticeEntry* g_batch_lattice = g_lattice + batch_index * element_count;

    if (global_index > element_count)
    {
        return;
    }

    // Loading feature vector whilst scaling to account for
    // the standard deviation introduced by the blur stage.
    float features[FEATURE_COUNT];

    for (int i = 0; i < FEATURE_COUNT; i++)
    {
        features[i] = g_batch_features[global_index + i * element_count];
        features[i] /= sqrt(2.0f / 3.0f) * (i + 1);
    }
    
    // Elevating feature vector to d+1 space
    float elevated_features[ELEVATED_COUNT];

    elevated_features[FEATURE_COUNT] = -FEATURE_COUNT * features[FEATURE_COUNT - 1];

    for (int i = FEATURE_COUNT - 1; i > 0; i--) 
    {
        elevated_features[i] = elevated_features[i + 1] - i * features[i - 1] + (i + 2) * features[i];
    }

    elevated_features[0] = elevated_features[1] + 2 * features[0];

    // Finding closest zero-colored lattice point
    int closest_lattice_point[ELEVATED_COUNT];
    float diff_from_point[ELEVATED_COUNT];
    int sum = 0;

    for (int i = 0; i < ELEVATED_COUNT; i++) 
    {
        int coord = round(elevated_features[i] / ELEVATED_COUNT);

        closest_lattice_point[i] = coord * ELEVATED_COUNT;
        diff_from_point[i] = elevated_features[i] - coord * ELEVATED_COUNT;

        sum += coord;
    }

    // Sorting differential to find the permutation between this simplex and the canonical one
    int rank[ELEVATED_COUNT];

    for (int i = 0; i < ELEVATED_COUNT; i++) 
    {
        rank[i] = 0;

        for (int j = 0; j < ELEVATED_COUNT; j++) 
        {
            float diff_i = diff_from_point[i];
            float diff_j = diff_from_point[j];

            if (diff_i < diff_j || (diff_i == diff_j && i > j)) 
            {
                rank[i]++;
            }
        }
    }

    // Ensure point lies on the plane
    for (int i = 0; i < ELEVATED_COUNT; i++) 
    {
        rank[i] += sum;

        int offset = ((rank[i] < 0) - (rank[i] >= ELEVATED_COUNT)) * ELEVATED_COUNT;

        diff_from_point[i] -= offset;
        closest_lattice_point[i] += offset;
        rank[i] += offset;
    }

    // Finding barycentric coordinates
    float barycentrics[ELEVATED_COUNT + 1];

    for (int i = 0; i < ELEVATED_COUNT + 1; i++) 
    {
        barycentrics[i] = 0;
    }

    for (int i = 0; i < ELEVATED_COUNT; i++) 
    {
        float delta = diff_from_point[i] / ELEVATED_COUNT;
        barycentrics[ELEVATED_COUNT - 1 - rank[i]] += delta;
        barycentrics[ELEVATED_COUNT - rank[i]] -= delta;
    }

    barycentrics[0] += 1.0f + barycentrics[ELEVATED_COUNT];

    // Compute the location of the lattice point explicitly (all but
    // the last coordinate - it's redundant because they sum to zero)
    LatticeEntry lattice_entry;

    for (int i = 0; i < ELEVATED_COUNT; i++) 
    {
        lattice_entry.element_index = global_index;
        lattice_entry.barycentric_weight = barycentrics[i];

        for (int j = 0; j < FEATURE_COUNT; j++) 
        {
            lattice_entry.lattice_coordinate[j] = closest_lattice_point[j] + i;

            if (rank[j] > FEATURE_COUNT - i) 
            {
                lattice_entry.lattice_coordinate[j] -= ELEVATED_COUNT;
            }
        }

        g_batch_lattice[global_index + i * element_count] = lattice_entry;
    }
}

__global__ void splat(const float* g_inputs, const float* g_barycentrics, const uint* g_element_indices, float* g_contributions, const uint element_count)
{

}

__global__ static void blur(const float* inputs, const float* barrycentrics, const uint* indices, float* contributions)
{

}

void phl_cuda(const float* inputs, const float* features, float* output, const uint batch_count, const uint element_count)
{
    dim3 grid(TILE(element_count, 16), batch_count);
    dim3 block(16, 1);

    LatticeEntry* lattice = (LatticeEntry*)output;

    create_lattice<<<grid, block>>>(features, lattice, element_count);

    LatticeEntry* lattice_host = new LatticeEntry[element_count * ELEVATED_COUNT * sizeof(LatticeEntry)];

    cudaMemcpy(lattice_host, lattice, element_count * ELEVATED_COUNT * sizeof(LatticeEntry), cudaMemcpyDeviceToHost);

    for (int i = 0; i < element_count * FEATURE_COUNT; i++)
    {
        py::print(
            "element: ",
            lattice_host[i].element_index,

            "\tcoord: ",
            lattice_host[i].lattice_coordinate[0],
            lattice_host[i].lattice_coordinate[1],
            
            "\tweight: ",
            lattice_host[i].barycentric_weight
        );
    }
}
