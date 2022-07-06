import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Set

from weak_nlp.shared import common_util


def get_token_range(df_record: pd.DataFrame) -> Set[int]:
    """Given some row has a `chunk_idx_start = 4` and `chunk_idx_end = 6`, this calculates the range between,
    i.e., {4, 5, 6}

    Args:
        df_record (pd.DataFrame): DataFrame with chunk idxs

    Returns:
        Set[int]: Set with the range
    """
    df_record["range"] = df_record.apply(
        lambda x: list(range(x["chunk_idx_start"], x["chunk_idx_end"] + 1)),
        axis=1,
    )
    token_lists = df_record["range"].tolist()
    token_set = set([item for sublist in token_lists for item in sublist])
    return token_set


def flatten_range_df(df: pd.DataFrame) -> pd.DataFrame:
    """Converts a dataframe in chunk idx format into a flattened format, in which each token is listed as a separate row.

    Args:
        df (pd.DataFrame): DataFrame with chunk idx format

    Returns:
        pd.DataFrame: DataFrame containing each token in the range as a row
    """
    df["range"] = df.apply(
        lambda x: list(range(x["chunk_idx_start"], x["chunk_idx_end"] + 1)),
        axis=1,
    )

    df_concat_ranged = []
    for _, row in df.iterrows():
        for row_idx, token_idx in enumerate(row.range):
            row_ranged = {
                "record": row.record,
                "label": row.label,
                "confidence": row.confidence,
                "token": token_idx,
                "beginner": row_idx == 0,
            }
            if "source" in row.keys():
                row_ranged["source"] = row.source
            df_concat_ranged.append(row_ranged)
    return pd.DataFrame(df_concat_ranged)


def add_conflicts_and_overlaps(
    quantity: Dict[str, Dict[str, int]],
    label: str,
    df_noisy_vectors_sub_record_label: pd.DataFrame,
    df_noisy_vectors_without_source_sub_record: pd.DataFrame,
    estimation_factor: int,
):
    """For a given noisy vector filtered down to record and label groupings, this calculates the overlaps
    and conflicts to a set of given other source vectors defined as a DataFrame

    Args:
        quantity (Dict[str, Dict[str, int]]): Containing the existing quantity metrics; also the return value
        label (str): Current label of interest
        df_noisy_vectors_sub_record_label (pd.DataFrame): Current source vector of interest on record and label grouping
        df_noisy_vectors_without_source_sub_record (pd.DataFrame): Containing all other noisy vectors without the current source vector of interest on record level
        estimation_factor (int): Multiplier given that computation is expensive and only a sample size is used to estimate overlaps and conflicts

    Returns:
        _type_: Containing the existing quantity metrics
    """
    for _, row in df_noisy_vectors_sub_record_label.iterrows():
        if any(
            [
                idx
                in df_noisy_vectors_without_source_sub_record.loc[
                    df_noisy_vectors_without_source_sub_record["label"] != label
                ]["token"].tolist()
                for idx in row.range
            ]
        ):
            quantity[label]["source_conflicts"] += 1
        if any(
            [
                idx
                in df_noisy_vectors_without_source_sub_record.loc[
                    df_noisy_vectors_without_source_sub_record["label"] == label
                ]["token"].tolist()
                for idx in row.range
            ]
        ):
            quantity[label]["source_overlaps"] += 1
    quantity[label]["source_conflicts"] *= estimation_factor
    quantity[label]["source_overlaps"] *= estimation_factor
    return quantity


def add_noisy_label_chunks_to_merged(
    merged_noisy_label_chunks: List[Dict[str, Any]],
    label: str,
    df_quartets_sub_label: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Adds noisy and potentially intersected labels to a cleansed and merged label chunk list

    Args:
        merged_noisy_label_chunks (List[Dict[str, Any]]): List of merged label chunks
        label (str): Current label of interest
        df_quartets_sub_label (pd.DataFrame): Containing noisy and potentially intersected labels

    Returns:
        List[Dict[str, Any]]: List of merged label chunks
    """
    df_quartets_sub_label_next = df_quartets_sub_label.shift(-1)
    new_token = True
    for (idx, row), (_, row_next) in zip(
        df_quartets_sub_label.iterrows(),
        df_quartets_sub_label_next.iterrows(),
    ):
        if idx < len(df_quartets_sub_label) - 1:
            if new_token:
                merged_token_set = row.token_set.copy()
                confs = [row.confidence]
                new_token = False

            if len(row.token_set.intersection(row_next.token_set)) > 0:
                merged_token_set.update(row_next.token_set)
                confs.append(row_next.confidence)
            else:
                merged_noisy_label_chunks.append(
                    {
                        "label": label,
                        "token_set": merged_token_set,
                        "confidence": max(confs),
                    }
                )
                new_token = True
        else:
            if new_token:
                merged_token_set = row.token_set.copy()
                confs = [row.confidence]
            merged_noisy_label_chunks.append(
                {
                    "label": label,
                    "token_set": merged_token_set,
                    "confidence": max(confs),
                }
            )
    return merged_noisy_label_chunks


def _ensemble(row: pd.Series, c: int, k: int) -> List[Tuple[str, float, int, int]]:
    """Integrates all relevant data from a given noisy label matrix row into a list of weakly supervised extraction tags

    Args:
        row (_type_): Single row from a DataFrame
        c (int): slope of the function
        k (int): what input should yield 0.5 probability?

    Returns:
        List[Tuple[str, float, int, int]]: List of weakly supervised extraction tags
    """
    quartets = []
    for column in row.keys():
        quartets.extend(row[column])
    df_quartets = pd.DataFrame(
        quartets,
        columns=["label", "confidence", "token_set", "chunk_idx_start"],
    )
    merged_noisy_label_chunks = []
    for label, df_quartets_sub_label in df_quartets.groupby("label"):
        df_quartets_sub_label = df_quartets_sub_label.sort_values(
            by="chunk_idx_start"
        ).reset_index(drop=True)

        merged_noisy_label_chunks = add_noisy_label_chunks_to_merged(
            merged_noisy_label_chunks, label, df_quartets_sub_label
        )

    preds = []
    if len(merged_noisy_label_chunks) > 0:
        df_noisy_label_chunks = pd.DataFrame(merged_noisy_label_chunks)
        df_noisy_label_chunks = df_noisy_label_chunks.sort_values(
            by="confidence", ascending=False
        )

        delete_idxs_dict = defaultdict(list)
        for idx, row in df_noisy_label_chunks.iterrows():
            for other_idx, other_row in df_noisy_label_chunks.drop(idx).iterrows():
                if len(row.token_set.intersection(other_row.token_set)) > 0:
                    if other_idx not in delete_idxs_dict.keys():
                        delete_idxs_dict[idx].append(other_idx)
        delete_idxs_list = [
            item for sublist in delete_idxs_dict.values() for item in sublist
        ]
        df_noisy_label_chunks_filtered = df_noisy_label_chunks.drop(delete_idxs_list)

        for _, row in df_noisy_label_chunks_filtered.iterrows():
            label = row["label"]
            confidence = row["confidence"]
            tokens = row["token_set"]
            if confidence > 0:
                confidence = common_util.sigmoid(confidence, c=c, k=k)
                pred = label, confidence, min(tokens), max(tokens)
                preds.append(pred)
    return preds
