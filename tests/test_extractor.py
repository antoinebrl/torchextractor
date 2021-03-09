import unittest
from collections import OrderedDict

import torch
from torch import nn

from torchextractor.extractor import Extractor


class MyTinyVGG(nn.Module):
    def __init__(self):
        super(MyTinyVGG, self).__init__()
        in_channels = 3
        nb_channels = 12
        nb_classes = 17

        self.block1 = self._make_layer(in_channels, nb_channels)
        in_channels, nb_channels = nb_channels, 2 * nb_channels
        self.block2 = self._make_layer(in_channels, nb_channels)
        in_channels, nb_channels = nb_channels, 2 * nb_channels
        self.block3 = self._make_layer(in_channels, nb_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(nb_channels, nb_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(nb_channels, nb_channels),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(nb_channels, nb_classes),
        )

    def _make_layer(self, in_channels, nb_channels):
        layer1 = nn.Sequential(
            nn.Conv2d(in_channels, nb_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nb_channels),
            nn.ReLU(inplace=True),
        )
        layer2 = nn.Sequential(
            nn.Conv2d(nb_channels, nb_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(nb_channels),
            nn.ReLU(inplace=True),
        )
        return nn.Sequential(
            OrderedDict([("layer1", layer1), ("layer2", layer2), ("pool", nn.MaxPool2d(kernel_size=2, stride=2))])
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.squeeze(3).squeeze(2)
        x = self.classifier(x)
        return x


class TestExtractor(unittest.TestCase):
    def test_model_output(self):
        model = MyTinyVGG()
        extractor = Extractor(model, ["block1", "block2"]).eval()
        input = torch.rand((5, 3, 32, 32))

        output_model = model(input)
        output_extractor, _ = extractor(input)

        self.assertTrue(torch.allclose(output_model, output_extractor))

    def test_forward_capture_feature_maps(self):
        model = MyTinyVGG()
        names = ["block1", "block2"]
        model = Extractor(model, names).eval()

        input = torch.rand((5, 3, 32, 32))
        output, feature_maps = model(input)

        self.assertTrue(all(True if name in feature_maps else False for name in names))
        self.assertEqual(list(feature_maps["block1"].shape), [5, 12, 16, 16])
        self.assertEqual(list(feature_maps["block2"].shape), [5, 24, 8, 8])

    def test_capture_latest_feature_map(self):
        model = MyTinyVGG()
        names = ["block1", "block2"]
        extractor = Extractor(model, names).eval()

        input1 = torch.rand((5, 3, 32, 32))
        model(input1)
        feature_maps1 = extractor.collect()
        shapes = {name: list(output.shape) for name, output in feature_maps1.items()}

        input2 = torch.rand((5, 3, 64, 64))
        model(input2)
        feature_maps2 = extractor.collect()

        self.assertTrue(all(True if name in feature_maps1 else False for name in names))
        for name in names:
            self.assertNotEqual(shapes[name], list(feature_maps2[name].shape))
            self.assertEqual(list(feature_maps1[name].shape), list(feature_maps2[name].shape))

    def test_destroy_extractor(self):
        model = MyTinyVGG()
        names = ["block1", "block2"]
        extractor = Extractor(model, names).eval()

        input1 = torch.rand((5, 3, 32, 32))
        model(input1)
        feature_maps = extractor.collect()
        shapes1 = {name: list(output.shape) for name, output in feature_maps.items()}

        del extractor

        input2 = torch.rand((5, 3, 64, 64))
        model(input2)
        shapes2 = {name: list(output.shape) for name, output in feature_maps.items()}

        # captured content should not change if hooks are no longer operating
        for name in names:
            self.assertEqual(shapes1[name], shapes2[name])

    def test_onnx_export(self):
        model = MyTinyVGG()
        names = ["block1", "block2"]
        model = Extractor(model, names).eval()
        input = torch.rand((5, 3, 32, 32))
        output, feature_maps = model(input)

        torch.onnx.export(model, input, "/tmp/model.onnx", output_names=["classifier"] + list(feature_maps.keys()))

        try:
            import onnx

            model = onnx.load("/tmp/model.onnx")
            output_names = [node.name for node in model.graph.output]
            for name in names:
                self.assertTrue(name in output_names)
        except ImportError as e:
            print(e)


if __name__ == "__main__":
    unittest.main()
