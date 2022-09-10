from collections import defaultdict
from typing import Optional, Tuple
import pandas as pd

from weak_nlp.shared import common_util


def _ensemble_single(row: pd.Series, c: int, k: int) -> Optional[Tuple[str, float]]:
    """Integrates all relevant data from a given noisy label matrix row into one weakly supervised classification

    Args:
        row (pd.Series): Single row from a DataFrame
        c (int): slope of the function
        k (int): what input should yield 0.5 probability?

    Returns:
        Optional[Tuple[str, float]]: Weakly supervised label and confidence; If confidence <= 0, this returns None.
    """
    voters = defaultdict(float)
    for column in row.keys():
        pair_or_empty = row[column]
        if pair_or_empty != "-":
            label_name, confidence = pair_or_empty
            voters[label_name] += confidence

    max_voter = max(voters, key=voters.get)  # e.g. clickbait
    sum_votes = sum(list(voters.values()))
    max_vote = voters[max_voter]

    confidence = max_vote - (sum_votes - max_vote)
    if confidence > 0:
        confidence = common_util.sigmoid(confidence, c=c, k=k)
        return max_voter, confidence


def _ensemble_multi(row: pd.Series, c: int, k: int, correlations: pd.DataFrame):
    voters = defaultdict(float)
    for column in row.keys():
        for pair in row[column]:
            label_name, confidence = pair
            voters[label_name] += confidence
    label_candidates = list(voters.keys())
    for label_name, confidence in voters.items():
        confidence = common_util.sigmoid(confidence, c=c, k=k)
        for label_candidate in label_candidates:
            if label_candidate != label_name:
                factor = correlations[label_name][label_candidate]
                confidence *= 1 + factor
        voters[label_name] = confidence
    return [
        [label_name, confidence]
        for label_name, confidence in voters.items()
        if confidence > 0.5
    ]
