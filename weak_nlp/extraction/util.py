import pandas as pd
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Set

from weak_nlp.shared import common_util


def get_token_range(df_record: pd.DataFrame) -> Set[int]:
    df_record["range"] = df_record.apply(
        lambda x: list(range(x["chunk_idx_start"], x["chunk_idx_end"] + 1)),
        axis=1,
    )
    token_lists = df_record["range"].tolist()
    token_set = set([item for sublist in token_lists for item in sublist])
    return token_set


def flatten_range_df(df, include_source=True):
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
            if include_source:
                row_ranged["source"] = row.source
            df_concat_ranged.append(row_ranged)
    return pd.DataFrame(df_concat_ranged)


def add_conflicts_and_overlaps(
    quantity,
    label,
    df_noisy_vectors_sub_record_label,
    df_noisy_vectors_without_source_sub_record,
    estimation_factor,
):
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


def add_noisy_label_chunks(
    merged_noisy_label_chunks, label, df_quartets_sub_label
) -> List[Dict[str, Any]]:
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


def _ensemble(row) -> List[Tuple[str, float, int, int]]:
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

        merged_noisy_label_chunks = add_noisy_label_chunks(
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
                confidence = common_util.sigmoid(confidence)
                pred = label, confidence, min(tokens), max(tokens)
                preds.append(pred)
    return preds
