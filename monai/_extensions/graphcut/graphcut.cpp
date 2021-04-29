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

torch::Tensor graphcut(torch::Tensor input_tensor, torch::Tensor weights_tensor)
{
    torch::Tensor output_tensor = torch::empty_like(input_tensor.narrow(1, 0, 1));

    uint sizes[DIMENSION_COUNT];
    uint element_count = input_tensor.stride(1);

    for (int i=0; i < DIMENSION_COUNT; i++)
    {
        sizes[i] = input_tensor.size(i+2);
    }

    float* input = input_tensor.data_ptr<float>();
    float* weights = weights_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();

    graphcut_cuda(input, weights, output, sizes, element_count);

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("graphcut", torch::wrap_pybind_function(graphcut));
}
