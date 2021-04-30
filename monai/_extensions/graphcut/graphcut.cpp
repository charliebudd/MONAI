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

#include "graphcut.h"

using namespace torch::indexing;

torch::Tensor graphcut(torch::Tensor image_tensor, torch::Tensor weights_tensor)
{
    uint batch_count = image_tensor.size(0);
    uint element_count = image_tensor.stride(1);
    uint sizes[DIMENSION_COUNT];
    uint strides[DIMENSION_COUNT];

    for (int i=0; i < DIMENSION_COUNT; i++)
    {
        sizes[i] = image_tensor.size(i+2);
        strides[i] = image_tensor.stride(i+2);
    }

    int dim = image_tensor.dim();
    long int* output_size = new long int[dim];
    memcpy(output_size, image_tensor.sizes().data(), dim * sizeof(long int));
    output_size[1] = CONNECTION_COUNT;
    torch::Tensor output_tensor = torch::empty(c10::IntArrayRef(output_size, dim), torch::dtype(image_tensor.dtype()).device(image_tensor.device()));
    delete output_size;

    // torch::Tensor output_tensor = torch::empty_like(image_tensor.narrow(1, 0, 1).expand({-1, CONNECTION_COUNT, -1, -1})).cuda();

    float* image = image_tensor.data_ptr<float>();
    float* weights = weights_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();

    graphcut_cuda(image, weights, output, batch_count, element_count, sizes, strides);

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("graphcut", torch::wrap_pybind_function(graphcut));
}
