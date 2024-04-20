"""Baseline submission for the job description classification challenge."""

from scripts import base, registry


@registry.register()
class Baseline(base.PromptSubmission):
    """Baseline submission."""

    def build_prompt(self, job_description: str) -> str:
        """Builds a prompt for classification of job description."""
        prompt = f"""
        
        Say "YES" if the given job description is suitable for
        a freshers other wise say "NO".

        {job_description}
        
        """
        return prompt.strip()

    def parse_response(self, model_response: str) -> bool:
        """Parses a response from the LLM to decide the final answer.

        Args:
            model_response: Output of the llm for the given prompt.

        Returns:
            True is the job_description is for a fresher otherwise False.
        """
        model_response = model_response.lower()
        if "yes" in model_response:
            return True
        return False
