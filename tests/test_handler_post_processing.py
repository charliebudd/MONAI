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

import unittest

import torch
from parameterized import parameterized

from monai.engines import SupervisedEvaluator
from monai.handlers import PostProcessing
from monai.transforms import Activationsd, AsDiscreted, Compose, CopyItemsd

# test lambda function as `transform`
TEST_CASE_1 = [{"transform": lambda x: dict(pred=x["pred"] + 1.0)}, torch.tensor([[[[1.9975], [1.9997]]]])]
# test composed post transforms as `transform`
TEST_CASE_2 = [
    {
        "transform": Compose(
            [
                CopyItemsd(keys="filename", times=1, names="filename_bak"),
                AsDiscreted(keys="pred", threshold_values=True, to_onehot=True, n_classes=2),
            ]
        )
    },
    torch.tensor([[[[1.0], [1.0]], [[0.0], [0.0]]]]),
]


class TestHandlerPostProcessing(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_compute(self, input_params, expected):
        data = [
            {"image": torch.tensor([[[[2.0], [3.0]]]]), "filename": "test1"},
            {"image": torch.tensor([[[[6.0], [8.0]]]]), "filename": "test2"},
        ]
        # set up engine, PostProcessing handler works together with post_transform of engine
        engine = SupervisedEvaluator(
            device=torch.device("cpu:0"),
            val_data_loader=data,
            epoch_length=2,
            network=torch.nn.PReLU(),
            post_transform=Compose([Activationsd(keys="pred", sigmoid=True)]),
            val_handlers=[PostProcessing(**input_params)],
        )
        engine.run()

        torch.testing.assert_allclose(engine.state.output["pred"], expected)
        filename = engine.state.output.get("filename_bak")
        if filename is not None:
            self.assertEqual(filename, "test2")


if __name__ == "__main__":
    unittest.main()
