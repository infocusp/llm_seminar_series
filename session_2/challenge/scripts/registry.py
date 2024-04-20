"""Registry of all the submitted prompts."""

from typing import Type

from scripts import base

_SUBMISSIONS_REGISTRY: dict[str, Type[base.PromptSubmission]] = {}


def register():
    """Returns a decorator that registers a submission with its file as key."""

    def _register(klass: Type[base.PromptSubmission]):
        name = klass.__module__.split(".")[-1]
        _SUBMISSIONS_REGISTRY[name] = klass
        return klass

    return _register


def get(name: str) -> base.PromptSubmission:
    """Returns the submission registered with the given name."""
    if name not in _SUBMISSIONS_REGISTRY:
        raise NotImplementedError(f"Submission with name {name} not found.")
    klass = _SUBMISSIONS_REGISTRY[name]
    return klass()


def get_all() -> list[Type[base.PromptSubmission]]:
    """Returns all the submissions."""
    return list(_SUBMISSIONS_REGISTRY.values())
