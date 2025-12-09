"""
Submodule explaining.explainers - contains explainer classes and the ExplainerFactory.
"""

import importlib
import pkgutil

# Import all submodules under this package
for module_info in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{module_info.name}")
