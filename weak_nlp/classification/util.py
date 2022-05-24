from collections import defaultdict
from typing import Optional, Tuple

from weak_nlp.shared import common_util


def _ensemble(row) -> Optional[Tuple[str, float]]:
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
        confidence = common_util.sigmoid(confidence)
        return max_voter, confidence
