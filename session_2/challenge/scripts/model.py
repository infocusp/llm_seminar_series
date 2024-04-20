"""Model inference."""

import g4f  # noqa: F401


class Model:
    """Base class for LLM."""

    def generate(self, prompt: str) -> str:
        """Returns a generation for prompt."""
        return ""


class G4fModel(Model):
    """A free gpt4 model.

    Reference: https://github.com/xtekky/gpt4free
    """

    def generate(self, prompt: str) -> str:
        """Completes a prompt using gpt-4 for free model."""
        # response = g4f.ChatCompletion.create(
        #     model=g4f.models.gpt_4,
        #     messages=[{"role": "user", "content": prompt}],
        # )
        response = "yes"
        return response
