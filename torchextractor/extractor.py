from functools import partial
from typing import Any, Callable, Dict
from typing import Iterable as IterableType
from typing import List, Tuple

from torch import nn

from .naming import attach_name_to_modules


def hook_capture_module_output(module: nn.Module, input: Any, output: Any, feature_maps: Dict[str, Any]):
    """
    Hook function to capture the output of the module.

    Parameters
    ----------
    module: nn.Module
        The module doing the computations.
    input:
        Whatever is provided to the module.
    output:
        Whatever is computed by the module.
    feature_maps: dictionary - keys: fully qualifying module names
        Placeholder to store the output of the modules so it can be used later on
    """
    feature_maps[module._extractor_fullname] = output


def register_hook(module_filter_fn: Callable, hook: Callable, hook_handles: List) -> Callable:
    """
    Attach a hook to some relevant modules.

    Parameters
    ----------
    module_filter_fn: callable
        A filtering function called for each module. When evaluated to `True` a hook is registered.
    hook: callable
        The hook to register. See documentation about PyTorch hooks.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    hook_handles: list
        A placeholders containing all newly registered hooks

    Returns
    -------
        callable function to apply on each module

    """

    def init_hook(module: nn.Module):
        if module_filter_fn(module, module._extractor_fullname):
            handle = module.register_forward_hook(hook)
            hook_handles.append(handle)

    return init_hook


class Extractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        module_names: IterableType[str] = None,
        module_filter_fn: Callable = None,
        caching_fn: Callable = None,
    ):
        """
        Capture the intermediate feature maps of of model.
        Behave like torch.nn.Module.

        Parameters
        ----------
        model: nn.Module,
            The model to extract features from.

        module_names: list of str, default None
            The fully qualified names of the modules producing the relevant feature maps.

        module_filter_fn: callable, default None
            A filtering function. Takes a module and module name as input and returns True for modules
            producing the relevant features. Either `module_names` or `module_filter_fn` should be
            provided but not both at the same time.

            Example::

                def module_filter_fn(module, name):
                    return isinstance(module, torch.nn.Conv2d)

        caching_fn: callable, default None
            Operation to carry at each forward pass.
            If not None, :func:`forward <collector.FeatureMapExtractor.forward>`
            and :func:`collect <collector.FeatureMapExtractor.collect>` return an empty dictionary
        """
        assert (
            module_names is not None or module_filter_fn is not None
        ), "Module names or a filtering function must be provided"
        assert not (module_names is not None and module_filter_fn is not None), (
            "You should either specify the fully qualifying names of the modules or a filtering function "
            "but not both at the same time"
        )

        super(Extractor, self).__init__()
        self.model = attach_name_to_modules(model)

        self.feature_maps = {}
        self.hook_handles = []

        module_filter_fn = module_filter_fn or (lambda module, name: name in module_names)
        caching_fn = caching_fn or hook_capture_module_output
        caching_fn = partial(caching_fn, feature_maps=self.feature_maps)
        self.model.apply(register_hook(module_filter_fn, caching_fn, self.hook_handles))

    def collect(self) -> Dict[str, nn.Module]:
        """
        Returns the structure holding the most recent feature maps.

        Notes
        _____
            The return structure is mutated at each forward pass of the model.
            It is the caller responsibility to duplicate the structure content if needed.
        """
        return self.feature_maps

    def clear_placeholder(self):
        """
        Resets the structure holding captured feature maps.
        """
        self.feature_maps.clear()

    def forward(self, *args, **kwargs) -> Tuple[Any, Dict[str, nn.Module]]:
        """
        Performs model computations and collects feature maps

        Returns
        -------
            Model output and intermediate feature maps
        """
        output = self.model(*args, **kwargs)
        return output, self.feature_maps

    def __del__(self):
        # Unregister hooks
        for handle in self.hook_handles:
            handle.remove()
