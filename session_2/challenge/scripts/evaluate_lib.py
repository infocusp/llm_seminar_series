"""Library function for evaluating a prompt on a particular dataset."""

import logging

import tqdm
from scripts import model, registry
from submissions import *  # noqa: F401, F403
from submissions import baseline, baseline_copy  # noqa: F401


def evaluate(dataset: list[tuple[str, bool]], prompt_name: str):
    """Evaluates the prompt submission."""
    # Loads a free gpt4 model.
    llm = model.G4fModel()

    # Loads a prompt submission.
    prompt_handler = registry.get(name=prompt_name)

    # Generate results for the dataset.
    correct_pred = 0
    for idx, (job_description, target) in enumerate(tqdm.tqdm(dataset)):
        prompt = prompt_handler.build_prompt(job_description=job_description)
        response = llm.generate(prompt=prompt)
        prediction = prompt_handler.parse_response(model_response=response)
        if prediction == target:
            correct_pred += 1
            result = "[PASS]"
        else:
            result = "[FAIL]"

        logging.debug(
            "No=%d. target=%s prediction=%s %s\n[prompt]\n%s\n[response]\n%s"
            % (idx, target, prediction, result, prompt, response)
        )
    acc = correct_pred / len(dataset) * 100
    return acc
