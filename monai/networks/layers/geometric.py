# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from monai.utils.module import optional_import

_C, _ = optional_import("monai._C")

__all__ = ["BuildGraph", "GraphCut"]

class BuildGraph(torch.autograd.Function):
    """
    Constructs a connected graph for use in geometric operations.

    See:

    Args:
        input_tensor (torch.Tensor): input tensor.

    Returns:
        output_graph (torch.Tensor): output graph.
    """

    @staticmethod
    def forward(ctx, input_tensor):
        output_graph = _C.calculate_edge_weights(input_tensor)
        return output_graph


class GraphCut(torch.autograd.Function):
    """
    Segments the input graph based on the max-flow-min-cut theorem.

    See:
        https://en.wikipedia.org/wiki/Graph_cuts_in_computer_vision

    Args:
        input_graph (torch.Tensor): input graph.

    Returns:
        output (torch.Tensor): NOT SURE.
    """

    @staticmethod
    def forward(ctx, input_graph, source_weights, sink_weights):
        output_segmentation =  _C.graph_cut(input_graph, source_weights, sink_weights)
        return output_segmentation
