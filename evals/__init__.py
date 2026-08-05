"""
Evaluation harnesses for SFR.

Run these as modules from the project root so that `app` is importable:

    python -m evals.simple_eval     # local, no API key needed
    python -m evals.sfr_eval        # LangSmith dataset + LLM-as-judge

Running them as file paths (python evals/simple_eval.py) puts evals/ on sys.path
instead of the project root, and `import app` then fails.
"""
