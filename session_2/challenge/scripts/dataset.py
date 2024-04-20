"""Utilities to load evaluation datasets."""

import glob
import os


def load_sample_test_set(samples_dir: str) -> list[tuple[str, bool]]:
    """Loads sample job descriptions and answers for local testing."""
    sample_files = glob.glob(os.path.join(samples_dir, "*.txt"))
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
