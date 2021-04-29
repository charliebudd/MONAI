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

from monai._extensions.loader import load_module

__all__ = ["Graphcut"]

class Graphcut():
    """
    Uses initial weights and the min-flow/max-cut theorem to optimize the boundry between two reigions of an image.
    """

    def __init__(self, channel_count, spatial_dimensions, use_diagonals):
        """
        Args:
                channel_count (int): The number of channels per image element.
                spatial_dimensions (int): The spatial dimension of the images to be processed.
                use_diagonals (bool): Whether to allow diagonal conections.
        """
        self.channel_count = channel_count
        self.spatial_dimensions = spatial_dimensions
        self.use_diagonals = use_diagonals
        self.compiled_extention = load_module(
            "graphcut", {"CHANNEL_COUNT": channel_count, "DIMENSION_COUNT": spatial_dimensions, "USE_DIAGONALS": "true" if use_diagonals else "false"}, True
        )

    def apply(self, input_tensor, weights_tensor):
        assert input_tensor.size(1) == self.channel_count, f"Expected a channel count of {self.channel_count}"
        assert input_tensor.dim() - 2 == self.spatial_dimensions, f"Expected a spatial dimension of {self.spatial_dimensions}"
        return _GraphcutFunc.apply(input_tensor, weights_tensor, self.compiled_extention)


class _GraphcutFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weights_tensor, compiled_extention):
        return compiled_extention.graphcut(input_tensor, weights_tensor)
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Graphcut does not support backpropagation")
