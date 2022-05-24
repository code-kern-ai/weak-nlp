import pandas as pd
import numpy as np
from weak_nlp import NoisyLabelMatrix


def get_all_noisy_vectors_df(nlm: NoisyLabelMatrix):
    dfs = []
    for vector in nlm.vectors_noisy:
        df = pd.DataFrame(vector.associations)
        df["source"] = vector.identifier
        dfs.append(df)
    return pd.concat(dfs)


def calc_precision(row):
    denominator = row["true_positives"] + row["false_positives"]
    if denominator == 0:
        return 0.0
    else:
        return row["true_positives"] / denominator


def calc_recall(row):
    denominator = row["true_positives"] + row["false_negatives"]
    if denominator == 0:
        return 0.0
    else:
        return row["true_positives"] / denominator


def sigmoid(x, c=1, k=1):
    # c: slope of the function
    # k: what input should yield 0.5 probability?
    return 1 / (1 + np.exp(-c * x + k))
