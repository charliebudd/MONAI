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

#include "gmm.h"
#include "gmm_cuda_linalg.cuh"

#define BLOCK_SIZE 32
#define TILE(SIZE, STRIDE) ((((SIZE) - 1)/(STRIDE)) + 1)

template<int warp_count, int load_count>
__global__ void CovarianceReductionKernel(int gaussian_index, const float* g_image, const int* g_alpha, float* g_matrices, int element_count)
{
    __shared__ float s_matrix_component[warp_count];

    const int block_size = warp_count << 5;

    int batch_index = blockIdx.z;

    const float* g_batch_image = g_image + batch_index * element_count * CHANNEL_COUNT;
    const int* g_batch_alpha = g_alpha + batch_index * element_count;
    float* g_batch_matrices = g_matrices + batch_index * GMM_COUNT * GMM_COMPONENT_COUNT * gridDim.x;

    int local_index = threadIdx.x;
    int block_index = blockIdx.x;
    int warp_index = local_index >> 5;
    int lane_index = local_index & 31;
    int global_index = local_index + block_index * block_size * load_count;
    int matrix_offset = (gaussian_index * gridDim.x + block_index) * GMM_COMPONENT_COUNT;

    float matrix[MATRIX_COMPONENT_COUNT];

    for (int i = 0; i < MATRIX_COMPONENT_COUNT; i++)
    {
        matrix[i] = 0;
    }

    for (int load = 0; load < load_count; load++)
    { 
        global_index += load * block_size;

        if (global_index < element_count)
        { 
            int my_alpha = g_batch_alpha[global_index];
    
            if (my_alpha != -1)
            {
                if (gaussian_index == (my_alpha & 15) + (my_alpha >> 4) * MIXTURE_COUNT)
                {
                    float feature[CHANNEL_COUNT + 1];

                    feature[0] = 1;

                    for (int i = 0; i < CHANNEL_COUNT; i++)
                    {
                        feature[i + 1] = g_batch_image[global_index + i * element_count] * 255;
                    }

                    for (int index = 0, i = 0; i < CHANNEL_COUNT + 1; i++)
                    {
                        for (int j = i; j < CHANNEL_COUNT + 1; j++, index++)
                        {
                            matrix[index] += feature[i] * feature[j];
                        }
                    }
                }
            }
        }
    }

    __syncthreads();

    for (int i = 0; i < MATRIX_COMPONENT_COUNT; i++)
    {
        float matrix_component = matrix[i];

        matrix_component += __shfl_down_sync(0xffffffff, matrix_component, 16);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  8);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  4);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  2);
        matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  1);

        if (lane_index == 0)
        {
            s_matrix_component[warp_index] = matrix_component;
        }

        __syncthreads();

        if (warp_index == 0) 
        { 
            matrix_component = s_matrix_component[lane_index];

            if (warp_count >= 32) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component, 16); }
            if (warp_count >= 16) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  8); }
            if (warp_count >=  8) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  4); }
            if (warp_count >=  4) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  2); }
            if (warp_count >=  2) { matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  1); }

            if (lane_index == 0)
            {
                g_batch_matrices[matrix_offset + i] = matrix_component;
            }
        }

        __syncthreads();
    }
}

template<int block_size, bool invert_matrix>
__global__ void CovarianceFinalizationKernel(const float* g_matrices, float* g_gmm, int matrix_count)
{
    __shared__ float s_matrix_component[block_size];
    __shared__ float s_gmm[GMM_COMPONENT_COUNT];

    int batch_index = blockIdx.z;

    const float* g_batch_matrices = g_matrices + batch_index * GMM_COUNT * GMM_COMPONENT_COUNT * matrix_count;
    float* g_batch_gmm = g_gmm + batch_index * GMM_COUNT * GMM_COMPONENT_COUNT;

    int local_index = threadIdx.x;
    int gmm_index = blockIdx.x;
    int matrix_offset = gmm_index * matrix_count;
    
    int load_count = TILE(matrix_count, block_size);

    float norm_factor = 1.0f;

    for (int index = 0, i = 0; i < CHANNEL_COUNT + 1; i++)
    {
        for (int j = i; j < CHANNEL_COUNT + 1; j++, index++)
        {
            float matrix_component = 0.0f;

            for(int load = 0; load < load_count; load++)
            {
                int matrix_index = local_index + load * block_size;

                if(matrix_index < matrix_count)
                {
                    matrix_component += g_batch_matrices[(matrix_offset + matrix_index) * GMM_COMPONENT_COUNT + index];
                }
            }

            s_matrix_component[local_index] = matrix_component; __syncthreads();

            if (block_size >= 512) { if (local_index < 256) { s_matrix_component[local_index] += s_matrix_component[local_index + 256]; } __syncthreads(); }
            if (block_size >= 256) { if (local_index < 128) { s_matrix_component[local_index] += s_matrix_component[local_index + 128]; } __syncthreads(); }
            if (block_size >= 128) { if (local_index <  64) { s_matrix_component[local_index] += s_matrix_component[local_index +  64]; } __syncthreads(); }
            if (block_size >=  64) { if (local_index <  32) { s_matrix_component[local_index] += s_matrix_component[local_index +  32]; } __syncthreads(); }

            if (local_index <  32)
            { 
                matrix_component = s_matrix_component[local_index];

                matrix_component += __shfl_down_sync(0xffffffff, matrix_component, 16);
                matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  8);
                matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  4);
                matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  2);
                matrix_component += __shfl_down_sync(0xffffffff, matrix_component,  1);

                if (local_index == 0)
                {
                    float constant = i == 0 ? 0.0f : s_gmm[i] * s_gmm[j];

                    if (i != 0 && i == j)
                    {
                        constant += 1.0e-3f;
                    }

                    s_gmm[index] = norm_factor * matrix_component - constant;

                    if (index == 0 && matrix_component > 0)
                    {
                        norm_factor = 1.0f / matrix_component;
                    }
                }
            }

            __syncthreads();
        }
    }

    float* matrix = s_gmm + (CHANNEL_COUNT + 1);
    float* det_ptr = s_gmm + MATRIX_COMPONENT_COUNT;

    CalculateDeterminant(matrix, det_ptr, local_index);

    if (invert_matrix)
    {
        InvertMatrix(matrix, *det_ptr, local_index);
    }

    if (local_index < GMM_COMPONENT_COUNT)
    {
        g_batch_gmm[gmm_index * GMM_COMPONENT_COUNT + local_index] = s_gmm[local_index];
    }
}

// Single block, 32xMIXTURE_COUNT
__global__ void GMMcommonTerm(float *g_gmm)
{
    int batch_index = blockIdx.z;

    float* g_batch_gmm = g_gmm + batch_index * GMM_COUNT * GMM_COMPONENT_COUNT;

    int gmm_index = (threadIdx.x * MIXTURE_COUNT) + threadIdx.y;

    float gmm_n = threadIdx.x < MIXTURE_SIZE ? g_batch_gmm[gmm_index * GMM_COMPONENT_COUNT] : 0.0f;

    float sum = gmm_n;

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum,  8);
    sum += __shfl_down_sync(0xffffffff, sum,  4);
    sum += __shfl_down_sync(0xffffffff, sum,  2);
    sum += __shfl_down_sync(0xffffffff, sum,  1);

    if (threadIdx.x < MIXTURE_SIZE)
    {
        float det = g_batch_gmm[gmm_index * GMM_COMPONENT_COUNT + MATRIX_COMPONENT_COUNT];
        float commonTerm =  gmm_n / (sqrtf(det) * sum);

        g_batch_gmm[gmm_index * GMM_COMPONENT_COUNT + MATRIX_COMPONENT_COUNT] = commonTerm;
    }
}

__device__ float GMMTerm(float* feature, const float *gmm)
{
    const float* average_feature = gmm + 1;
    const float* matrix = gmm + CHANNEL_COUNT + 1;

    float diff[CHANNEL_COUNT];

    for (int i = 0; i < CHANNEL_COUNT; i++)
    {
        diff[i] = feature[i] - average_feature[i];
    }

    float value = 0.0f;

    for (int index = 0, i = 0; i < CHANNEL_COUNT; i++)
    {
        for (int j = i; j < CHANNEL_COUNT; j++, index++)
        {
            float term = diff[i] * diff[j] * matrix[index];

            value += i == j ? term : 2 * term;
        }
    }

    return gmm[MATRIX_COMPONENT_COUNT] * expf(-0.5f * value);
}

__global__ void GMMDataTermKernel(const float *image, const float *gmm, float* output, int element_count)
{
    int batch_index = blockIdx.z;

    const float* g_batch_image = image + batch_index * element_count * CHANNEL_COUNT;
    const float* g_batch_gmm = gmm + batch_index * GMM_COUNT * GMM_COMPONENT_COUNT;
    float* g_batch_output = output + batch_index * element_count * MIXTURE_COUNT;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= element_count) return;

    float feature[CHANNEL_COUNT];

    for (int i = 0; i < CHANNEL_COUNT; i++)
    {
        feature[i] = g_batch_image[index + i * element_count] * 255;
    }

    float weights[MIXTURE_COUNT];
    float weight_total = 0.0f;

    for(int i = 0; i < MIXTURE_COUNT; i++)
    {
        float mixture_weight = 0.0f;

        for(int j = 0; j < MIXTURE_SIZE; j++)
        {
            mixture_weight += GMMTerm(feature, &g_batch_gmm[(MIXTURE_COUNT * j + i) * GMM_COMPONENT_COUNT]);
        }

        weights[i] = mixture_weight;
        weight_total += mixture_weight;
    }

    for(int i = 0; i < MIXTURE_COUNT; i++)
    {
        g_batch_output[index + i * element_count] = weights[i] / weight_total;
    }
}

struct GMMSplit_t
{
    int idx;
    float threshold;
    float eigenvector[CHANNEL_COUNT];
};

// 1 Block, 32xMIXTURE_COUNT
__global__ void GMMFindSplit(GMMSplit_t *gmmSplit, int gmmK, float *gmm)
{
    int batch_index = blockIdx.z;

    float* g_batch_gmm = gmm + batch_index * GMM_COUNT * GMM_COMPONENT_COUNT;
    GMMSplit_t* g_batch_gmmSplit = gmmSplit + batch_index * MIXTURE_COUNT;

    int gmm_idx = threadIdx.x * MIXTURE_COUNT + threadIdx.y;

    float eigenvalue = 0;
    float eigenvector[CHANNEL_COUNT];

    if (threadIdx.x < gmmK)
    {
        float* matrix = g_batch_gmm + gmm_idx * GMM_COMPONENT_COUNT + (CHANNEL_COUNT + 1);
        largest_eigenpair(matrix, eigenvector, &eigenvalue);
    }

    float max_value = eigenvalue;

    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value, 16));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value,  8));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value,  4));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value,  2));
    max_value = max(max_value, __shfl_xor_sync(0xffffffff, max_value,  1));

    if (max_value == eigenvalue)
    {
        GMMSplit_t split;

        float* average_feature = gmm + gmm_idx * GMM_COMPONENT_COUNT + 1;

        split.idx = threadIdx.x;
        split.threshold = scalar_prod(average_feature, eigenvector);

        for (int i = 0; i < CHANNEL_COUNT; i++)
        {
            split.eigenvector[i] = eigenvector[i];
        }

        g_batch_gmmSplit[threadIdx.y] = split;
    }
}

#define DO_SPLIT_DEGENERACY 4

__global__ void GMMDoSplit(const GMMSplit_t *gmmSplit, int k, const float *image, int *alpha, int element_count)
{
    __shared__ GMMSplit_t s_gmmSplit[MIXTURE_COUNT];

    int batch_index = blockIdx.z;

    const GMMSplit_t* g_batch_gmmSplit = gmmSplit + batch_index * MIXTURE_COUNT;
    const float* g_batch_image = image + batch_index * element_count * CHANNEL_COUNT;
    int* g_batch_alpha = alpha + batch_index * element_count;

    int *s_linear = (int *) s_gmmSplit;
    int *g_linear = (int *) g_batch_gmmSplit;

    if (threadIdx.x < MIXTURE_COUNT * sizeof(GMMSplit_t))
    {
        s_linear[threadIdx.x] = g_linear[threadIdx.x];
    }

    __syncthreads();

    int index = threadIdx.x + blockIdx.x * BLOCK_SIZE * DO_SPLIT_DEGENERACY;

    for (int i = 0; i < DO_SPLIT_DEGENERACY; i++)
    {
        index += BLOCK_SIZE;

        if (index < element_count)
        {
            int my_alpha = g_batch_alpha[index];

            if(my_alpha != -1)
            {
                int select = my_alpha & 15;
                int gmm_idx = my_alpha >> 4;
    
                if (gmm_idx == s_gmmSplit[select].idx)
                {
                    // in the split cluster now
                    float feature[CHANNEL_COUNT];

                    for (int i = 0; i < CHANNEL_COUNT; i++)
                    {
                        feature[i] = g_batch_image[index + i * element_count] * 255;
                    }
                    
                    float value = scalar_prod(s_gmmSplit[select].eigenvector, feature);
    
                    if (value > s_gmmSplit[select].threshold)
                    {
                        // assign pixel to new cluster
                        g_batch_alpha[index] =  k + select;
                    }
                }
            }
        }
    }
}

#define THREADS 512
#define WARPS 16
#define BLOCK (WARPS << 5)
#define LOAD 4

void GMMInitialize(const float *image, int *alpha, float *gmm, float *scratch_mem, int batch_count, int element_count)
{
    int block_count = TILE(element_count, BLOCK * LOAD);

    float* block_gmm_scratch = scratch_mem;
    GMMSplit_t* gmm_split_scratch = (GMMSplit_t*) scratch_mem;

    int gmm_N = MIXTURE_COUNT * MIXTURE_SIZE;

    for (int k = MIXTURE_COUNT; k < gmm_N; k+=MIXTURE_COUNT)
    {
        for (int i = 0; i < k; ++i)
        {
            CovarianceReductionKernel<WARPS, LOAD><<<{block_count, 1, batch_count}, BLOCK>>>(i, image, alpha, block_gmm_scratch, element_count);
        }

        CovarianceFinalizationKernel<THREADS, false><<<{k, 1, batch_count}, THREADS>>>(block_gmm_scratch, gmm, block_count);

        GMMFindSplit<<<{1, 1, batch_count}, dim3(BLOCK_SIZE, MIXTURE_COUNT)>>>(gmm_split_scratch, k / MIXTURE_COUNT, gmm);
        GMMDoSplit<<<{TILE(element_count, BLOCK_SIZE * DO_SPLIT_DEGENERACY), 1, batch_count}, BLOCK_SIZE>>>(gmm_split_scratch, (k / MIXTURE_COUNT) << 4, image, alpha, element_count);
    }
}

void GMMUpdate(const float *image, int *alpha, float *gmm, float *scratch_mem, int batch_count, int element_count)
{
    int block_count = TILE(element_count, BLOCK * LOAD);

    float* block_gmm_scratch = scratch_mem;

    int gmm_N = MIXTURE_COUNT * MIXTURE_SIZE;

    for (int i = 0; i < gmm_N; ++i)
    {
        CovarianceReductionKernel<WARPS, LOAD><<<{block_count, 1, batch_count}, BLOCK>>>(i, image, alpha, block_gmm_scratch, element_count);
    }

    CovarianceFinalizationKernel<THREADS, true><<<{gmm_N, 1, batch_count}, THREADS>>>(block_gmm_scratch, gmm, block_count);

    GMMcommonTerm<<<{1, 1, batch_count}, dim3(BLOCK_SIZE, MIXTURE_COUNT)>>>(gmm);
}

void GMMDataTerm(const float *image, const float *gmm, float* output, int batch_count, int element_count)
{
    dim3 block(BLOCK_SIZE, 1);
    dim3 grid(TILE(element_count, BLOCK_SIZE), 1, batch_count);

    GMMDataTermKernel<<<grid, block>>>(image, gmm, output, element_count);
}

void GMM_Cuda(const float* input, const int* labels, float* output, int batch_count, int element_count)
{
    float* scratch_mem = output;
    float* gmm; 
    int* alpha;

    cudaMalloc(&gmm, batch_count * GMM_COUNT * GMM_COMPONENT_COUNT * sizeof(float));
    cudaMalloc(&alpha, batch_count * element_count * sizeof(int));

    cudaMemcpyAsync(alpha, labels, batch_count * element_count * sizeof(int), cudaMemcpyDeviceToDevice);
    
    GMMInitialize(input, alpha, gmm, scratch_mem, batch_count, element_count);
    GMMUpdate(input, alpha, gmm, scratch_mem, batch_count, element_count);
    GMMDataTerm(input, gmm, output, batch_count, element_count);

    cudaFree(alpha);
    cudaFree(gmm);
}