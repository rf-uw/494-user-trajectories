import importlib.util
import os


def load_scorer_schema(repo_path):
    candidates = [
        "sourcecode/scoring/constants.py",
        "scoring/constants.py",
        "scoring/src/scoring/constants.py",
    ]

    for rel in candidates:
        full = os.path.join(repo_path, rel)
        if os.path.exists(full):
            spec = importlib.util.spec_from_file_location("cn_constants", full)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError("Could not find constants.py in this commit")
