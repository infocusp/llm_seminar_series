"""Evaluates the submitted prompts.

You can copy session_2/challenge/submissions/baseline.py to modify your own
prompt and evaluate it locally using this script.

You need to pass the name used for registering a submission.

For example,

```
@registry.register("baseline")
class Baseline(base.PromptSubmission):

    def build_prompt(self, job_description: str) -> str:
        ...
```

In the above code, a Baseline class is registered with the name of `baseline`,
so you can run the below sample command to evaluate it.

python3 -m scripts.evaluate --prompt=baseline
"""

import glob
import logging
import os
from collections.abc import Sequence

import tqdm
from absl import app, flags
from scripts import model, registry
from submissions import baseline  # noqa: F401

_PROMPT = flags.DEFINE_string(
    "prompt", None, "Name of the prompt to evaluate."
)

_SAMPLES_DIR = "sample_inputs"


def load_sample_test_set() -> list[tuple[str, bool]]:
    """Loads sample job descriptions and answers for local testing."""
    sample_files = glob.glob(os.path.join(_SAMPLES_DIR, "*.txt"))
    sample_inputs = []
    for filepath in sample_files:
        content = open(filepath, "r").read()
        filename = os.path.basename(filepath).lower()
        if filename.endswith("_yes.txt"):
            target = True
        elif filename.endswith("_no.txt"):
            target = False
        else:
            raise ValueError(
                "File %s must end with yes.txt or no.txt" % filepath
            )
        target = True if "yes" in filename.lower() else False
        sample_inputs.append((content, target))
    return sample_inputs


def evaluate(prompt_name: str):
    """Evaluates the prompt submission."""
    # Loads a free gpt4 model.
    llm = model.G4fModel()

    # Loads a prompt submission.
    prompt_handler = registry.get(name=prompt_name)

    # Generate results for the dataset.
    dataset = load_sample_test_set()
    correct_pred = 0
    for job_description, target in tqdm.tqdm(dataset):
        prompt = prompt_handler.build_prompt(job_description=job_description)
        response = llm.generate(prompt=prompt)
        output = prompt_handler.parse_response(model_response=response)
        if output == target:
            correct_pred += 1

    logging.info("Acc : %.3f" % (correct_pred / len(dataset) * 100))


def main(argv: Sequence[str]) -> None:
    """Entrypoint."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    evaluate(prompt_name=_PROMPT.value)


if __name__ == "__main__":
    app.run(main)
