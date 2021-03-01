import unittest
from collections import OrderedDict

from torch import nn

from torchextractor.naming import attach_name_to_modules, find_modules_by_names


class TestNaming(unittest.TestCase):
    def test_sequential_model(self):
        model = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU())

        attach_name_to_modules(model)

        for i in range(4):
            self.assertEqual(model[i]._extractor_fullname, str(i))

    def test_sequential_model_with_dict(self):
        names = ["conv1", "relu1", "conv2", "relu2"]
        ops = [nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU()]
        model = nn.Sequential(OrderedDict(zip(names, ops)))

        attach_name_to_modules(model)

        for i, name in enumerate(names):
            self.assertEqual(model[i]._extractor_fullname, name)

    def test_sequential_module_inheritance(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

        model = MyModel()
        attach_name_to_modules(model)

        module_iter = model.children()
        module1 = next(module_iter)
        self.assertEqual(module1._extractor_fullname, "conv1")
        module2 = next(module_iter)
        self.assertEqual(module2._extractor_fullname, "conv2")
        # Only two modules
        self.assertIsNone(next(module_iter, None))

    def test_nested_modules(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Sequential(nn.Linear(4, 4), nn.Sigmoid(), nn.Linear(4, 1), nn.Sigmoid()),
                    nn.Sigmoid(),
                )

        model = MyModel()
        attach_name_to_modules(model)

        self.assertEqual(model.block1[0][2]._extractor_fullname, "block1.0.2")


class TestModuleSearch(unittest.TestCase):
    def test_sequential_model(self):
        names = ["conv1", "relu1", "conv2", "relu2"]
        ops = [nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU()]
        model = nn.Sequential(OrderedDict(zip(names, ops)))
        attach_name_to_modules(model)

        search_names = ["conv1", "relu2"]
        modules = find_modules_by_names(model, search_names)

        # All names have a match
        self.assertTrue(all([name in modules for name in search_names]))

        # Each name links to the right module
        self.assertEqual(id(modules["conv1"]), id(ops[0]))
        self.assertEqual(id(modules["relu2"]), id(ops[3]))

    def test_module_not_found(self):
        names = ["conv1", "relu1", "conv2", "relu2"]
        ops = [nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 64, 5), nn.ReLU()]
        model = nn.Sequential(OrderedDict(zip(names, ops)))
        attach_name_to_modules(model)

        search_names = ["conv1", "azertyuiop"]
        modules = find_modules_by_names(model, search_names)

        # Each name links to the right module
        self.assertFalse("azertyuiop" in modules)

    def test_nested_modules(self):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Sequential(nn.Linear(4, 4), nn.Sigmoid(), nn.Linear(4, 1), nn.Sigmoid()),
                    nn.Sigmoid(),
                )

        model = MyModel()
        attach_name_to_modules(model)
        modules = find_modules_by_names(model, ["block1.0.2"])

        self.assertEqual(id(model.block1[0][2]), id(modules["block1.0.2"]))


if __name__ == "__main__":
    unittest.main()
