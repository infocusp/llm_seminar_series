"""Base class for prompt submission."""

import abc


class PromptSubmission(abc.ABC):
    """Base class for prompt submission."""

    def __init__(self):
        """Initializes a prompt submission class."""
        pass

    @abc.abstractmethod
    def build_prompt(self, job_description: str) -> str:
        """Builds a prompt for classification of job description.

        Args:
            job_description: Input for classification.

        Returns:
            Input for the LLM.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parse_response(self, model_response: str) -> bool:
        """Parses a response from the LLM to decide the final answer.

        Args:
            model_response: Output of the llm for the given prompt.

        Returns:
            True is the job_description is for a fresher otherwise False.
        """
        raise NotImplementedError
