import logging
from collections.abc import Iterable
from typing import Dict
from typing import Iterable as IterableType
from typing import List

from torch import nn

logger = logging.getLogger(__name__)


def list_module_names(model: nn.Module) -> List[str]:
    """
    List names of modules and submodules.

    Parameters
    ----------
    model: nn.Module
        PyTorch model to examine.

    Returns
    -------
    List of names
    """
    return [name for name, module in model.named_modules()]


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


def find_modules_by_names(model: nn.Module, names: IterableType[str]) -> Dict[str, nn.Module]:
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
        dict: name -> module
            If no match is found for a name, it is not added to the returned structure

    """
    assert isinstance(names, (list, tuple))
    found_modules = {name: module for name, module in model.named_modules() if name in names}
    return found_modules
