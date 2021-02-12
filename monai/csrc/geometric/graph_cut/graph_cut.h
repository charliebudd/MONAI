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

#pragma once

#include <torch/extension.h>

torch::Tensor GraphCut_Cuda(torch::Tensor input_graph, torch::Tensor source_weights, torch::Tensor sink_weights, int iterations, int a, int b);

torch::Tensor GraphCut(torch::Tensor input_graph, torch::Tensor source_weights, torch::Tensor sink_weights, int iterations, int a, int b) {
    return GraphCut_Cuda(input_graph, source_weights, sink_weights, iterations, a, b);
}
