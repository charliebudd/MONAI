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

#include "phl.h"

torch::Tensor phl(torch::Tensor input_tensor, torch::Tensor feature_tensor)
{
    c10::DeviceType device_type = input_tensor.device().type();

    uint batch_count = input_tensor.size(0);
    uint element_count = input_tensor.stride(1);

    torch::Tensor output_tensor = torch::empty_like(input_tensor);
    output_tensor = output_tensor.narrow(1, 0, 1).expand({-1, ELEVATED_COUNT, -1, -1}).contiguous();

    float* inputs = input_tensor.data_ptr<float>();
    float* features = feature_tensor.data_ptr<float>();
    float* outputs = output_tensor.data_ptr<float>();

    if(device_type == torch::kCUDA)
    {
        phl_cuda(inputs, features, outputs, batch_count, element_count);
    }
    else
    {
        phl_cpu(inputs, features, outputs, batch_count, element_count);
    }

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("phl", torch::wrap_pybind_function(phl));
}
