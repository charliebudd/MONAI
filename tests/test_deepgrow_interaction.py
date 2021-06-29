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

from monai.apps.deepgrow.interaction import Interaction
from monai.data import Dataset
from monai.engines import SupervisedTrainer
from monai.engines.utils import IterationEvents
from monai.transforms import Activationsd, Compose, ToNumpyd


def add_one(engine):
    if engine.state.best_metric is -1:
        engine.state.best_metric = 0
    else:
        engine.state.best_metric = engine.state.best_metric + 1


class TestInteractions(unittest.TestCase):
    def run_interaction(self, train, compose):
        data = []
        for i in range(5):
            data.append({"image": torch.tensor([float(i)]), "label": torch.tensor([float(i)])})
        network = torch.nn.Linear(1, 1)
        lr = 1e-3
        opt = torch.optim.SGD(network.parameters(), lr)
        loss = torch.nn.L1Loss()
        dataset = Dataset(data, transform=None)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=5)

        iteration_transforms = [Activationsd(keys="pred", sigmoid=True), ToNumpyd(keys="pred")]
        iteration_transforms = Compose(iteration_transforms) if compose else iteration_transforms

        i = Interaction(transforms=iteration_transforms, train=train, max_interactions=5)
        self.assertEqual(len(i.transforms.transforms), 2, "Mismatch in expected transforms")

        # set up engine
        engine = SupervisedTrainer(
            device=torch.device("cpu"),
            max_epochs=1,
            train_data_loader=data_loader,
            network=network,
            optimizer=opt,
            loss_function=loss,
            iteration_update=i,
        )
        engine.add_event_handler(IterationEvents.INNER_ITERATION_STARTED, add_one)
        engine.add_event_handler(IterationEvents.INNER_ITERATION_COMPLETED, add_one)

        engine.run()
        self.assertIsNotNone(engine.state.batch.get("probability"), "Probability is missing")
        self.assertEqual(engine.state.best_metric, 9)

    def test_train_interaction(self):
        self.run_interaction(train=True, compose=True)

    def test_val_interaction(self):
        self.run_interaction(train=False, compose=False)


if __name__ == "__main__":
    unittest.main()
