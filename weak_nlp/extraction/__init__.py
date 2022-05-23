import weak_nlp
import numpy as np
import pandas as pd
from collections import defaultdict


def sigmoid(x, c=1, k=1):
    # c: slope of the function
    # k: what input should yield 0.5 probability?
    return 1 / (1 + np.exp(-c * x + k))


class ExtractionAssociation(weak_nlp.Association):
    def __init__(self, record, label, chunk_idx_start, chunk_idx_end, confidence=1):
        super().__init__(record, label, confidence)
        self.chunk_idx_start = chunk_idx_start
        self.chunk_idx_end = chunk_idx_end


class ENLM(weak_nlp.NoisyLabelMatrix):
    def __init__(self, vectors):
        super().__init__(vectors)

    def quality_metrics(self):
        if self.vector_reference is None:
            raise Exception(
                "Can't calculate the quality metrics without reference vector"
            )

        for idx, vector_noisy in enumerate(self.vectors_noisy):

            df_reference = self.vector_reference.associations
            df_noisy = vector_noisy.associations

            quality = {}
            reference_labels = df_reference["label"].dropna().unique()
            for label_name in reference_labels:
                quality[label_name] = {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                }
            for (record, label), df_reference_grouped in df_reference.groupby(
                ["record", "label"]
            ):
                df_noisy_grouped = df_noisy.loc[
                    (df_noisy["record"] == record) & (df_noisy["label"] == label)
                ].copy()

                df_reference_grouped["range"] = df_reference_grouped.apply(
                    lambda x: list(range(x["chunk_idx_start"], x["chunk_idx_end"] + 1)),
                    axis=1,
                )
                token_list_reference = df_reference_grouped["range"].tolist()
                token_set_reference = set(
                    [item for sublist in token_list_reference for item in sublist]
                )

                if len(df_noisy_grouped) > 0:
                    df_noisy_grouped["range"] = df_noisy_grouped.apply(
                        lambda x: list(
                            range(x["chunk_idx_start"], x["chunk_idx_end"] + 1)
                        ),
                        axis=1,
                    )
                    token_list_noisy = df_noisy_grouped["range"].tolist()
                    token_set_noisy = set(
                        [item for sublist in token_list_noisy for item in sublist]
                    )

                    true_positives = len(
                        token_set_reference.intersection(token_set_noisy)
                    )
                    false_positives = len(
                        token_set_noisy.difference(token_set_reference)
                    )
                    false_negatives = len(
                        token_set_reference.difference(token_set_noisy)
                    )
                else:
                    true_positives = 0
                    false_positives = 0
                    false_negatives = len(token_set_reference)
                quality[label]["true_positives"] += true_positives
                quality[label]["false_positives"] += false_positives
                quality[label]["false_negatives"] += false_negatives
            self.vectors_noisy[idx].quality = quality.copy()

        statistics = []
        for vector_noisy in self.vectors_noisy:
            vector_stats = {"identifier": vector_noisy.identifier}
            for label_name in vector_noisy.quality.keys():
                vector_stats["label_name"] = label_name
                quality = vector_noisy.quality[label_name]

                vector_stats["true_positives"] = quality["true_positives"]
                vector_stats["false_positives"] = quality["false_positives"]
                vector_stats["false_negatives"] = quality["false_negatives"]
                statistics.append(vector_stats.copy())

        def calc_precision(row):
            sum_positives = row["true_positives"] + row["false_positives"]
            if sum_positives == 0:
                return 0.0
            else:
                return row["true_positives"] / sum_positives

        stats_df = pd.DataFrame(statistics)
        if len(stats_df) > 0:
            stats_df["precision"] = stats_df.apply(calc_precision, axis=1)

        return stats_df

    def quantity_metrics(self):
        dfs = []
        for vector in self.vectors_noisy:
            df = pd.DataFrame(vector.associations)
            df["source"] = vector.identifier
            dfs.append(df)
        df_concat = pd.concat(dfs)

        df_concat["range"] = df_concat.apply(
            lambda x: list(range(x["chunk_idx_start"], x["chunk_idx_end"] + 1)),
            axis=1,
        )

        df_concat_ranged = []
        for idx, row in df_concat.iterrows():
            for row_idx, token_idx in enumerate(row.range):
                df_concat_ranged.append(
                    {
                        "record": row.record,
                        "label": row.label,
                        "confidence": row.confidence,
                        "token": token_idx,
                        "beginner": row_idx == 0,
                        "source": row.source,
                    }
                )
        df_concat_ranged = pd.DataFrame(df_concat_ranged)

        for source, df_source in df_concat_ranged.groupby("source"):
            quantity = {
                label: {
                    "record_coverage": len(
                        df_source.loc[(df_source["label"] == label)].record.unique()
                    ),
                    "total_hits": df_source.loc[
                        (df_source["label"] == label)
                    ].beginner.sum(),
                    "source_overlaps": 0,
                    "source_conflicts": 0,
                }
                for label in df_source["label"].dropna().unique()
            }

            for (record, label), df_source in df_concat.loc[
                df_concat["source"] == source
            ].groupby(["record", "label"]):
                df_others = df_concat_ranged.loc[
                    (df_concat_ranged["source"] != source)
                    & (df_concat_ranged["record"] == record)
                ]
                for idx, row in df_source.iterrows():
                    if any(
                        [
                            idx
                            in df_others.loc[df_others["label"] != label][
                                "token"
                            ].tolist()
                            for idx in row.range
                        ]
                    ):
                        quantity[label]["source_conflicts"] += 1
                    if any(
                        [
                            idx
                            in df_others.loc[df_others["label"] == label][
                                "token"
                            ].tolist()
                            for idx in row.range
                        ]
                    ):
                        quantity[label]["source_overlaps"] += 1

            for idx, vector in enumerate(self.vectors_noisy):
                if vector.identifier == source:
                    self.vectors_noisy[idx].quantity = quantity.copy()

        statistics = []
        for vector_noisy in self.vectors_noisy:
            vector_stats = {"identifier": vector_noisy.identifier}
            for label_name in vector_noisy.quantity.keys():
                vector_stats["label_name"] = label_name

                quantity = vector_noisy.quantity[label_name]
                vector_stats["record_coverage"] = quantity["record_coverage"]
                vector_stats["total_hits"] = quantity["total_hits"]
                vector_stats["source_conflicts"] = quantity["source_conflicts"]
                vector_stats["source_overlaps"] = quantity["source_overlaps"]
                statistics.append(vector_stats.copy())

        stats_df = pd.DataFrame(statistics)
        return stats_df

    def weakly_supervise(self):
        stats_df = self.quality_metrics()
        if len(stats_df) == 0:
            raise Exception("Empty statistics; can't compute weak supervision")
        stats_lkp = stats_df.set_index(["identifier", "label_name"]).to_dict(
            orient="index"
        )  # pairwise [heuristic, label] lookup for precision

        cnlm_df = pd.DataFrame(self.records, columns=["record"])
        cnlm_df = cnlm_df.set_index("record")

        for vector in self.vectors_noisy:
            vector_df = vector.associations
            vector_df["prediction"] = vector_df.apply(
                lambda x: [
                    x["label"],
                    stats_lkp[(vector.identifier, x["label"])]["precision"]
                    * (x["confidence"]),
                    set(list(range(x["chunk_idx_start"], x["chunk_idx_end"] + 1))),
                    x["chunk_idx_start"],
                ],
                axis=1,
            )
            vector_series = vector_df.set_index("record")[["prediction"]]
            cnlm_df[vector.identifier] = np.empty((len(cnlm_df), 0)).tolist()
            for idx, row in vector_series.iterrows():
                cnlm_df[vector.identifier].loc[idx].append(row["prediction"])

        def ensemble(row):
            values = []
            for column in row.keys():
                values.extend(row[column])
            df = pd.DataFrame(
                values, columns=["label", "confidence", "token_set", "token_begin"]
            )
            merged_rows = []
            for label, df_label in df.groupby("label"):
                df_label = df_label.sort_values(by="token_begin").reset_index(drop=True)

                df_label_next = df_label.shift(-1)
                new_token = True
                for (idx, row), (_, row_next) in zip(
                    df_label.iterrows(), df_label_next.iterrows()
                ):
                    if idx < len(df_label) - 1:
                        if new_token:
                            merged_token_set = row.token_set.copy()
                            confs = [row.confidence]
                            new_token = False

                        if len(row.token_set.intersection(row_next.token_set)) > 0:
                            merged_token_set.update(row_next.token_set)
                            confs.append(row_next.confidence)
                        else:
                            merged_rows.append(
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
                        merged_rows.append(
                            {
                                "label": label,
                                "token_set": merged_token_set,
                                "confidence": max(confs),
                            }
                        )

            preds = []
            df = pd.DataFrame(merged_rows).sort_values(by="confidence", ascending=False)

            delete_idxs = defaultdict(list)
            for idx, row in df.iterrows():
                for other_idx, other_row in df.drop(idx).iterrows():
                    if len(row.token_set.intersection(other_row.token_set)) > 0:
                        if other_idx not in delete_idxs.keys():
                            delete_idxs[idx].append(other_idx)
            flat_list = [item for sublist in delete_idxs.values() for item in sublist]
            df = df.drop(flat_list)

            for _, row in df.iterrows():
                label = row["label"]
                confidence = row["confidence"]
                tokens = row["token_set"]
                if confidence > 0:
                    confidence = sigmoid(confidence)
                    pred = [label, confidence, min(tokens), max(tokens)]
                    preds.append(pred)
            return preds

        return cnlm_df.apply(ensemble, axis=1)
