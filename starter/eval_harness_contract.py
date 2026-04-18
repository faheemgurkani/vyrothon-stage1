"""
eval_harness_contract.py — Grader interface contract.
Do not modify the score() function signature or scoring rules.
"""

import json
import re
from typing import Any


def parse_tool_call(text: str) -> dict | None:
    """Extract and parse JSON from <tool_call>...</tool_call> tags."""
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def is_refusal(text: str) -> bool:
    """Return True if the output contains no tool call (plain-text refusal)."""
    return parse_tool_call(text) is None


def args_match(pred_args: dict, gold_args: dict, tol: float = 0.01) -> bool:
    """Check if all predicted args match ground-truth args within tolerance for numerics."""
    if set(pred_args.keys()) != set(gold_args.keys()):
        return False
    for key in gold_args:
        pv, gv = pred_args[key], gold_args[key]
        if isinstance(gv, (int, float)):
            try:
                if abs(float(pv) - float(gv)) / (abs(float(gv)) + 1e-9) > tol:
                    return False
            except (ValueError, TypeError):
                return False
        else:
            if str(pv).strip() != str(gv).strip():
                return False
    return True


def score(prediction: str, ground_truth: dict) -> float:
    """
    Score a single prediction against ground truth.

    ground_truth schema:
    {
        "expected_tool": str | None,   # None means refusal expected
        "expected_args": dict | None,  # None if refusal expected
        "is_refusal": bool
    }

    Returns:
        +1.0  correct tool + all args correct (numerics within ±1%)
        +0.5  correct tool + ≥1 arg wrong or missing
         0.0  wrong tool, malformed JSON, wrong refusal decision
        -0.5  emitted tool call when refusal was correct
    """
    expected_refusal: bool = ground_truth.get("is_refusal", False)
    pred_call = parse_tool_call(prediction)
    pred_is_refusal = pred_call is None

    if expected_refusal:
        if pred_is_refusal:
            return 1.0
        else:
            return -0.5

    if pred_is_refusal:
        return 0.0

    expected_tool = ground_truth.get("expected_tool")
    expected_args = ground_truth.get("expected_args", {})

    if pred_call.get("tool") != expected_tool:
        return 0.0

    pred_args = pred_call.get("args", {})
    if args_match(pred_args, expected_args):
        return 1.0
    else:
        return 0.5


def evaluate_dataset(predictions: list[str], dataset: list[dict]) -> dict[str, Any]:
    """
    Evaluate a list of predictions against the dataset.

    Each dataset entry must have keys: id, slice, ground_truth (dict as above).
    Returns per-slice scores and overall score.
    """
    assert len(predictions) == len(dataset), "Predictions and dataset length mismatch."

    slices: dict[str, list[float]] = {"A": [], "B": [], "C": [], "D": []}
    all_scores: list[float] = []

    for pred, entry in zip(predictions, dataset):
        s = score(pred, entry["ground_truth"])
        sl = entry.get("slice", "A")
        slices.setdefault(sl, []).append(s)
        all_scores.append(s)

    results = {
        "overall": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "total_score": sum(all_scores),
        "num_examples": len(all_scores),
        "per_slice": {k: (sum(v) / len(v) if v else None) for k, v in slices.items()},
        "per_slice_counts": {k: len(v) for k, v in slices.items()},
    }
    return results


# ---------------------------------------------------------------------------
# Grader entry point — do not rename or change the signature
# ---------------------------------------------------------------------------

def run_evaluation(run_fn, dataset_path: str) -> dict[str, Any]:
    """
    Load dataset from a .jsonl file and score the run_fn on it.

    run_fn must match: def run(prompt: str, history: list[dict]) -> str
    """
    dataset = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))

    predictions = []
    for entry in dataset:
        conversation = entry.get("conversation", [])
        if not conversation:
            predictions.append("")
            continue
        history = conversation[:-1]
        prompt = conversation[-1]["content"]
        predictions.append(run_fn(prompt, history))

    return evaluate_dataset(predictions, dataset)
