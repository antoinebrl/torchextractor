# torchextractor: PyTorch Intermediate Feature Extraction

## Introduction

Too many times some model definitions get remorselessly copy-pasted just because the
`forward` function does not return what the person expects. With `torchextractor`
you can easily capture intermediate feature maps and build whatever you fancy on 
top of it.

## Installation

```shell
pip install git+https://github.com/antoinebrl/torchextractor.git
```

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
