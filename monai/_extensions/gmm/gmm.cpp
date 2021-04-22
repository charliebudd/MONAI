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

#include "gmm.h"

py::tuple init()
{
    torch::Tensor gmm_tensor = torch::zeros({GMM_COUNT, GMM_COMPONENT_COUNT}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor scratch_tensor = torch::empty({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return py::make_tuple(gmm_tensor, scratch_tensor);
}

void learn(torch::Tensor gmm_tensor, torch::Tensor scratch_tensor, torch::Tensor feature_tensor, torch::Tensor weight_tensor)
{
    c10::DeviceType device_type = feature_tensor.device().type();

    unsigned int batch_count = feature_tensor.size(0);
    unsigned int element_count = feature_tensor.stride(1);

    unsigned int scratch_size = batch_count * (GMM_COUNT * element_count + GMM_COMPONENT_COUNT * GMM_COUNT * (element_count / (32 * 32)));

    if (scratch_tensor.size(0) < scratch_size)
    {
        scratch_tensor.resize_({scratch_size});
    }

    float* gmm = gmm_tensor.data_ptr<float>();
    float* scratch = scratch_tensor.data_ptr<float>();
    float* features = feature_tensor.data_ptr<float>();
    float* weights = weight_tensor.data_ptr<float>();

    if(device_type == torch::kCUDA)
    {
        learn_cuda(features, weights, gmm, scratch, batch_count, element_count);
    }
    else
    {
        learn_cpu(features, weights, gmm, scratch, batch_count, element_count);
    }
}

torch::Tensor apply(torch::Tensor gmm_tensor, torch::Tensor feature_tensor)
{
    c10::DeviceType device_type = feature_tensor.device().type();

    unsigned int dim = feature_tensor.dim();
    unsigned int batch_count = feature_tensor.size(0);
    unsigned int element_count = feature_tensor.stride(1);

    long int* output_size = new long int[dim];
    memcpy(output_size, feature_tensor.sizes().data(), dim * sizeof(long int));
    output_size[1] = MIXTURE_COUNT;
    torch::Tensor output_tensor = torch::empty(c10::IntArrayRef(output_size, dim), torch::dtype(torch::kFloat32).device(device_type));
    delete output_size;

    const float* gmm = gmm_tensor.data_ptr<float>();
    const float* features = feature_tensor.data_ptr<float>();
    float* output = output_tensor.data_ptr<float>();

    if(device_type == torch::kCUDA)
    {
        apply_cuda(gmm, features, output, batch_count, element_count);
    }
    else
    {
        apply_cpu(gmm, features, output, batch_count, element_count);
    }

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("init", torch::wrap_pybind_function(init), "Creates the parameters for a gaussian mixture model");
    m.def("learn", torch::wrap_pybind_function(learn), "Encorporates the provided data into the existing model");
    m.def("apply", torch::wrap_pybind_function(apply), "Applies the existing model to obtain class predictions");
}