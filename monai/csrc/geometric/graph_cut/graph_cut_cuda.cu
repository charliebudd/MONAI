#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "push_relabel.cuh"


torch::Tensor GraphCut_Cuda(torch::Tensor input_graph, torch::Tensor source_weights, torch::Tensor sink_weights, int iterations)
{
    PushRelabel push_relabel = PushRelabel(input_graph, source_weights, sink_weights);

    torch::Tensor output = push_relabel.Execute(iterations);

    return output;
}
