import logging
from collections.abc import Iterable
from typing import Dict
from typing import Iterable as IterableType

from torch import nn

logger = logging.getLogger(__name__)


def attach_name_to_modules(model: nn.Module) -> nn.Module:
    """
    Assign a unique name to each module based on the nested structure of the model.

    Parameters
    ----------
    model: nn.Module
        PyTorch model to decorate with fully qualifying names for each module.

    Returns
    -------
    model: nn.Module.
        The provided model as input.

    """
    for name, module in model.named_modules():
        module._extractor_fullname = name
    return model


def find_modules_by_names(model: nn.Module, names: IterableType[str]) -> Dict:
    """
    Find some modules given their fully qualifying names.

    Parameters
    ----------
    model: nn.Module
        PyTorch model to examine.
    names: list of str
        List of fully qualifying names.

    Returns
    -------
        dict: name -> output (often torch.Tensor)
            The provided names without any match will not appears in the returned dictionary

    """
    assert isinstance(names, Iterable)
    found_modules = {
        m._extractor_fullname: m
        for m in model.modules()
        if hasattr(m, "_extractor_fullname") and m._extractor_fullname in names
    }
    if len(found_modules) != len(names):
        logger.warning(
            "It looks like some names could not be find. "
            "Make sure `attach_name_to_modules(model)` is called before calling this search function"
        )
    return found_modules
