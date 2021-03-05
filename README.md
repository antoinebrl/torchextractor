# `torchextractor`: PyTorch Intermediate Feature Extraction

## Introduction

Too many times some model definitions get remorselessly copy-pasted just because the
`forward` function does not return what the person expects. You provide module names
and `torchextractor` takes care of the extraction for you.It's never been easier to
extract feature, add an extra loss or plug another head to a network.
Ler us know what amazing things you build with `torchextractor`!

## Installation

```shell
pip install git+https://github.com/antoinebrl/torchextractor.git
```

Requirements:
- Python >= 3.6+
- torch >= 1.4.0

## Usage

```python
import torch
import torchvision
import torchextractor as tx

model = torchvision.models.resnet18(pretrained=True)
model = tx.Extractor(model, ["layer1", "layer2", "layer3", "layer4"])
dummy_input = torch.rand(7, 3, 224, 224)
model_output, features = model(dummy_input)
feature_shapes = {name: f.shape for name, f in features.items()}
print(feature_shapes)

# {
#   'layer1': torch.Size([1, 64, 56, 56]),
#   'layer2': torch.Size([1, 128, 28, 28]),
#   'layer3': torch.Size([1, 256, 14, 14]),
#   'layer4': torch.Size([1, 512, 7, 7]),
# }
```

[See more examples](docs/source/examples.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/antoinebrl/torchextractor/HEAD?filepath=docs/source/examples.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antoinebrl/torchextractor/blob/master/docs/source/examples.ipynb)

## FAQ

**• How do I know the names of the modules?**

You can print all module names like this:
```python
for name, module in model.named_modules():
    print(name)
```

**• Why do some operations not get listed?**

It is not possible to add hooks if operations are not defined as modules.
Therefore, `F.relu` cannot be captured but `nn.Relu()` can.

**• How can I avoid listing all relevant modules?**

You can specify a custom filtering function to hook the relevant modules:
```python
# Hook everything !
module_filter_fn = lambda module, name: True

# Capture of all modules inside first layer
module_filter_fn = lambda module, name: name.startswith("layer1")

# Focus on all convolutions
module_filter_fn = lambda module, name: isinstance(module, torch.nn.Conv2d)

model = tx.Extractor(model, module_filter_fn=module_filter_fn)
```

**• Is it compatible with ONNX?**

`tx.Extractor` is compatible with ONNX! This means you can also access intermediate features maps after the export.

Pro-tip: name the output nodes by using `output_names` when calling `torch.onnx.export`.

**• Is it compatible with TorchScript?**

Bad news, TorchScript cannot take variable number of arguments and keyword-only arguments.
Good news, there is a workaround! The solution is to overwrite the `forward` function
of `tx.Extractor` to replicate the interface of the model.

```python
import torch
import torchvision
import torchextractor as tx

class MyExtractor(tx.Extractor):
    def forward(self, x1, x2, x3):
        # Assuming the model takes x1, x2 and x3 as input
        output = self.model(x1, x2, x3)
        return output, self.feature_maps

model = torchvision.models.resnet18(pretrained=True)
model = MyExtractor(model, ["layer1", "layer2", "layer3", "layer4"])
model_traced = torch.jit.script(model)
```

## Contributing

All feedbacks and contributions are welcomed. Feel free to report an issue or to create a pull request!

If you want to get hands-on:
1. (Fork and) clone the repo.
2. Create a virtual environment: `virtualenv -p python3 .venv && source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt && pip install -r requirements-dev.txt`
4. Hook auto-formatting tools: `pre-commit install`
5. Hack as much as you want!
6. Run tests: `python -m unittest discover -vs ./tests/`
7. Share your work and create a pull request.
