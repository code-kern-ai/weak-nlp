from collections import defaultdict
from typing import List, Optional
import pandas as pd
from weak_nlp import base
from weak_nlp.classification import ClassificationAssociation, ClassificationNLM
from weak_nlp.shared import common_util, exceptions


class MultilabelMulticlassAssociation(ClassificationAssociation):
    def __init__(self, record: str, labels: List[str], confidence: Optional[float] = 1):
        self.record = record
        self.labels = labels
        self.confidence = confidence


def _ensemble(row: pd.Series, c: int, k: int, correlations: pd.DataFrame):
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


class MultiLabelMultiClassNLM(ClassificationNLM):
    def __init__(self, vectors: base.SourceVector):
        super().__init__(vectors)

        df_rows = []
        for gt_label in self.vector_reference.associations.labels.tolist():
            row = {
                label_option: 0
                for label_option in self.vector_reference.associations.explode(
                    "labels"
                )["labels"]
                .dropna()
                .unique()
            }
            for label in gt_label:
                row[label] = 1
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        self.correlations = df.corr()

    def _set_quality_metrics_inplace(self) -> None:
        exploded_reference = self.vector_reference.associations.explode(
            "labels"
        ).dropna()
        for idx, vector_noisy in enumerate(self.vectors_noisy):
            if not vector_noisy.is_empty:
                exploded_noisy = vector_noisy.associations.explode("labels").dropna()
                df_inner_join = (
                    exploded_reference.set_index("record")
                    .join(
                        exploded_noisy.set_index("record"),
                        how="inner",
                        lsuffix="_reference",
                        rsuffix="_noisy",
                    )
                    .reset_index()
                )

                df_true_positives = df_inner_join.loc[
                    df_inner_join["labels_reference"] == df_inner_join["labels_noisy"]
                ].set_index(["record", "labels_reference"])

                df_false_negatives = df_inner_join.set_index(
                    ["record", "labels_reference"]
                )
                df_false_negatives = df_false_negatives.loc[
                    ~df_false_negatives.index.isin(df_true_positives.index)
                ]
                df_false_negatives = df_false_negatives.reset_index()[
                    ["record", "labels_reference"]
                ].drop_duplicates()

                df_false_positives = df_inner_join.set_index(["record", "labels_noisy"])
                df_false_positives = df_false_positives.loc[
                    ~df_false_positives.index.isin(df_true_positives.index)
                ]
                df_false_positives = df_false_positives.reset_index()[
                    ["record", "labels_noisy"]
                ].drop_duplicates()

                df_true_positives = df_true_positives.reset_index()[
                    ["record", "labels_reference"]
                ]

                quality = {}
                for label_name in (
                    exploded_reference["labels"].unique().tolist()
                    + exploded_noisy["labels"].unique().tolist()
                ):
                    quality[label_name] = {
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                    }

                for label_name in quality.keys():
                    quality[label_name]["true_positives"] = (
                        df_true_positives["labels_reference"] == label_name
                    ).sum()
                    quality[label_name]["false_positives"] = (
                        df_false_positives["labels_noisy"] == label_name
                    ).sum()
                    quality[label_name]["false_negatives"] = (
                        df_false_negatives["labels_reference"] == label_name
                    ).sum()
                self.vectors_noisy[idx].quality = quality.copy()

    def _set_quantity_metrics_inplace(self) -> None:
        df_noisy_vectors = common_util.get_all_noisy_vectors_df(self)
        for source, df_source in df_noisy_vectors.groupby("source"):
            quantity = {
                label: {
                    "record_coverage": len(
                        df_source.loc[df_source["labels"].apply(lambda x: label in x)]
                    ),
                    "source_overlaps": 0,
                    "source_conflicts": 0,
                }
                for label in df_source.explode("labels")["labels"].dropna().unique()
            }

            df_without_source = df_noisy_vectors.loc[
                (df_noisy_vectors["source"] != source)
                & (df_noisy_vectors["record"].isin(df_source.record.unique()))
                # no need to load parts of other heuristics we don't care about for this heuristic
            ]

            for record_series in df_without_source.groupby("record")["labels"]:
                record_id, labels = record_series
                labels = set([element for sublist in labels for element in sublist])
                row_source = set(
                    df_source.loc[df_source["record"] == record_id].iloc[0]["labels"]
                )
                if len(labels.intersection(row_source)) > 0:
                    for label in row_source:
                        quantity[label]["source_overlaps"] += 1

            for idx, vector in enumerate(self.vectors_noisy):
                if vector.identifier == source:
                    self.vectors_noisy[idx].quantity = quantity.copy()

    def weakly_supervise(self, c: Optional[int] = 7, k: Optional[int] = 3) -> pd.Series:
        stats_df = self.quality_metrics()
        if len(stats_df) == 0:
            raise exceptions.MissingStatsException(
                "Empty statistics; can't compute weak supervision"
            )
        stats_lkp = stats_df.set_index(["identifier", "label_name"]).to_dict(
            orient="index"
        )  # pairwise [heuristic, label] lookup for precision

        cnlm_df = pd.DataFrame(self.records, columns=["record"])
        cnlm_df = cnlm_df.set_index("record")
        for vector in self.vectors_noisy:
            vector_df = vector.associations
            if len(vector_df) > 0:
                vector_df["predictions"] = vector_df.apply(
                    lambda x: [
                        (
                            x["labels"][idx],
                            stats_lkp[(vector.identifier, x["labels"][idx])][
                                "precision"
                            ]
                            * (x["confidence"]),
                        )
                        for idx in range(len(x["labels"]))
                    ],
                    axis=1,
                )
                vector_series = vector_df.set_index("record")["predictions"]
                cnlm_df[vector.identifier] = vector_series
        return cnlm_df.apply(
            _ensemble, axis=1, c=1, k=1, correlations=self.correlations
        )
