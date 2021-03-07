API
===

Extractor
---------
.. autoclass:: torchextractor.Extractor(model: nn.Module, module_names: IterableType[str] = None, module_filter_fn: Callable = None, caching_fn: Callable = None)

Utils
-----

.. autofunction:: torchextractor.list_module_names

.. autofunction:: torchextractor.find_modules_by_names
