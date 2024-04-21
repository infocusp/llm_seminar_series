"""Registers all the submissions."""

import os

__all__ = []
for filename in os.listdir(os.path.dirname(os.path.realpath(__file__))):
    if filename.endswith(".py") and filename != "__init__.py":
        __all__.append(filename.split(".")[0])
