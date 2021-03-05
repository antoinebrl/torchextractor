API
===

Extractor
---------
.. autoclass:: torchextractor.Extractor(model: nn.Module, module_names: IterableType[str] = None, module_filter_fn: Callable = None, caching_fn: Callable = None)

Utils
-----

.. autofunction:: torchextractor.attach_name_to_modules

.. autofunction:: torchextractor.find_modules_by_names
