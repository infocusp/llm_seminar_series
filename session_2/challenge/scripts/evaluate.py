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

import logging
from collections.abc import Sequence

from absl import app, flags
from scripts import dataset, evaluate_lib

_PROMPT = flags.DEFINE_string(
    "prompt", None, "Name of the prompt to evaluate."
)

_DEBUG = flags.DEFINE_bool(
    "debug", True, "Prints prompt and response if true."
)


def evaluate_on_sample_dataset(prompt_name: str):
    """Evaluates the prompt on a sample_dataset."""
    sample_inputs = dataset.load_dataset_from_dir(samples_dir="dataset")
    acc = evaluate_lib.evaluate(dataset=sample_inputs, prompt_name=prompt_name)
    print("Accuracy: [%.3f] %%" % acc)  # noqa: T201


def main(argv: Sequence[str]) -> None:
    """Entrypoint."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if _DEBUG.value:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    evaluate_on_sample_dataset(prompt_name=_PROMPT.value)


if __name__ == "__main__":
    flags.mark_flag_as_required("prompt")
    app.run(main)
