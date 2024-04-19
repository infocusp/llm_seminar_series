"""Registry of all the submitted prompts."""

from typing import Type

from scripts import base

_SUBMISSIONS_REGISTRY: dict[str, Type[base.PromptSubmission]] = {}


def register(name: str):
    """Returns a decorator that registers a submission with the given name."""

    def _register(klass: Type[base.PromptSubmission]):
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
