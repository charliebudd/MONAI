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

__all__ = ["PHL"]

class PHL:

    def __init__(self, channel_count, feature_count):
        self.channel_count = channel_count
        self.feature_count = feature_count
        self.compiled_extention = load_module(
            "phl", {"CHANNEL_COUNT": channel_count, "FEATURE_COUNT": feature_count}, verbose_build=True
        )

    def apply(self, inputs, features):
        return self.compiled_extention.phl(inputs, features)
